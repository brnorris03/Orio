

program main

    implicit none
    
    integer, parameter :: double = selected_real_kind(10,40)
    integer, parameter :: single = selected_real_kind(5,20)
    
    real(double) :: orio_t_start, orio_t_end, orio_min_time, orio_delta_time
    integer      :: orio_i
    
#define N 520
#define NUMINTCELLS 512
#define HY_NUMXN 0
#include "decl_code.F90"


    
#include "init_code.F90"


    
    orio_min_time = X'7FF00000'   ! large number
    do orio_i = 1, ORIO_REPS
    
      orio_t_start = getClock()
    
       


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

  do i=5, numIntCells5-1, +2  
    aux(i)=sqrt(0.5*(game(i)-1)/game(i))
    ge=0.5*(hy_gmelft(i)+hy_gmergt(i))
    gc=0.5*(hy_gmclft(i)+hy_gmcrgt(i))
    gamfac(i)=(1-ge/gc)*(ge-1)
    gmin(i)=min(game(i-1),game(i),game(i+1))
    gmax(i)=max(game(i-1),game(i),game(i+1))
    aux((i+1))=sqrt(0.5*(game((i+1))-1)/game((i+1)))
    ge=0.5*(hy_gmelft((i+1))+hy_gmergt((i+1)))
    gc=0.5*(hy_gmclft((i+1))+hy_gmcrgt((i+1)))
    gamfac((i+1))=(1-ge/gc)*(ge-1)
    gmin((i+1))=min(game(i),game((i+1)),game(i+2))
    gmax((i+1))=max(game(i),game((i+1)),game(i+2))
  

  end do
  do i=(numIntCells5)-(MOD((numIntCells5), 2)), numIntCells5, +1  
    aux(i)=sqrt(0.5*(game(i)-1.0)/game(i))
    ge=0.5*(hy_gmelft(i)+hy_gmergt(i))
    gc=0.5*(hy_gmclft(i)+hy_gmcrgt(i))
    gamfac(i)=(1.0-ge/gc)*(ge-1.0)
    gmin(i)=min(game(i-1),game(i),game(i+1))
    gmax(i)=max(game(i-1),game(i),game(i+1))
  

  end do


  do i=5, numIntCells5-1, +2  
    pstar1(i)=hy_prght(i)-hy_plft(i)-hy_crght(i)*(hy_urght(i)-hy_ulft(i))
    pstar1(i)=hy_plft(i)+pstar1(i)*hy_clft(i)/(hy_clft(i)+hy_crght(i))
    pstar1(i)=max(hy_smallp,pstar1(i))
    pstar1((i+1))=hy_prght((i+1))-hy_plft((i+1))-hy_crght((i+1))*(hy_urght((i+1))-hy_ulft((i+1)))
    pstar1((i+1))=hy_plft((i+1))+pstar1((i+1))*hy_clft((i+1))/(hy_clft((i+1))+hy_crght((i+1)))
    pstar1((i+1))=max(hy_smallp,pstar1((i+1)))
  

  end do
  do i=(numIntCells5)-(MOD((numIntCells5), 2)), numIntCells5, +1  
    pstar1(i)=hy_prght(i)-hy_plft(i)-hy_crght(i)*(hy_urght(i)-hy_ulft(i))
    pstar1(i)=hy_plft(i)+pstar1(i)*(hy_clft(i)/(hy_clft(i)+hy_crght(i)))
    pstar1(i)=max(hy_smallp,pstar1(i))
  

  end do


  do i=5, numIntCells5-5, +6  
    gmstrl(i)=gamfac(i)*(pstar1(i)-hy_plft(i))
    gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar1(i)+hy_plft(i))
    gmstrr(i)=gamfac(i)*(pstar1(i)-hy_prght(i))
    gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar1(i)+hy_prght(i))
    gmstrl(i)=max(gmin(i),min(gmstrl(i),gmax(i)))
    gmstrr(i)=max(gmin(i),min(gmstrr(i),gmax(i)))
    gmstrl((i+1))=gamfac((i+1))*(pstar1((i+1))-hy_plft((i+1)))
    gmstrl((i+1))=hy_gmelft((i+1))+2*gmstrl((i+1))/(pstar1((i+1))+hy_plft((i+1)))
    gmstrr((i+1))=gamfac((i+1))*(pstar1((i+1))-hy_prght((i+1)))
    gmstrr((i+1))=hy_gmergt((i+1))+2*gmstrr((i+1))/(pstar1((i+1))+hy_prght((i+1)))
    gmstrl((i+1))=max(gmin((i+1)),min(gmstrl((i+1)),gmax((i+1))))
    gmstrr((i+1))=max(gmin((i+1)),min(gmstrr((i+1)),gmax((i+1))))
    gmstrl((i+2))=gamfac((i+2))*(pstar1((i+2))-hy_plft((i+2)))
    gmstrl((i+2))=hy_gmelft((i+2))+2*gmstrl((i+2))/(pstar1((i+2))+hy_plft((i+2)))
    gmstrr((i+2))=gamfac((i+2))*(pstar1((i+2))-hy_prght((i+2)))
    gmstrr((i+2))=hy_gmergt((i+2))+2*gmstrr((i+2))/(pstar1((i+2))+hy_prght((i+2)))
    gmstrl((i+2))=max(gmin((i+2)),min(gmstrl((i+2)),gmax((i+2))))
    gmstrr((i+2))=max(gmin((i+2)),min(gmstrr((i+2)),gmax((i+2))))
    gmstrl((i+3))=gamfac((i+3))*(pstar1((i+3))-hy_plft((i+3)))
    gmstrl((i+3))=hy_gmelft((i+3))+2*gmstrl((i+3))/(pstar1((i+3))+hy_plft((i+3)))
    gmstrr((i+3))=gamfac((i+3))*(pstar1((i+3))-hy_prght((i+3)))
    gmstrr((i+3))=hy_gmergt((i+3))+2*gmstrr((i+3))/(pstar1((i+3))+hy_prght((i+3)))
    gmstrl((i+3))=max(gmin((i+3)),min(gmstrl((i+3)),gmax((i+3))))
    gmstrr((i+3))=max(gmin((i+3)),min(gmstrr((i+3)),gmax((i+3))))
    gmstrl((i+4))=gamfac((i+4))*(pstar1((i+4))-hy_plft((i+4)))
    gmstrl((i+4))=hy_gmelft((i+4))+2*gmstrl((i+4))/(pstar1((i+4))+hy_plft((i+4)))
    gmstrr((i+4))=gamfac((i+4))*(pstar1((i+4))-hy_prght((i+4)))
    gmstrr((i+4))=hy_gmergt((i+4))+2*gmstrr((i+4))/(pstar1((i+4))+hy_prght((i+4)))
    gmstrl((i+4))=max(gmin((i+4)),min(gmstrl((i+4)),gmax((i+4))))
    gmstrr((i+4))=max(gmin((i+4)),min(gmstrr((i+4)),gmax((i+4))))
    gmstrl((i+5))=gamfac((i+5))*(pstar1((i+5))-hy_plft((i+5)))
    gmstrl((i+5))=hy_gmelft((i+5))+2*gmstrl((i+5))/(pstar1((i+5))+hy_plft((i+5)))
    gmstrr((i+5))=gamfac((i+5))*(pstar1((i+5))-hy_prght((i+5)))
    gmstrr((i+5))=hy_gmergt((i+5))+2*gmstrr((i+5))/(pstar1((i+5))+hy_prght((i+5)))
    gmstrl((i+5))=max(gmin((i+5)),min(gmstrl((i+5)),gmax((i+5))))
    gmstrr((i+5))=max(gmin((i+5)),min(gmstrr((i+5)),gmax((i+5))))
  

  end do
  do i=(numIntCells5)-(MOD((numIntCells5), 6)), numIntCells5, +1  
    gmstrl(i)=gamfac(i)*(pstar1(i)-hy_plft(i))
    gmstrl(i)=hy_gmelft(i)+2.0*gmstrl(i)/(pstar1(i)+hy_plft(i))
    gmstrr(i)=gamfac(i)*(pstar1(i)-hy_prght(i))
    gmstrr(i)=hy_gmergt(i)+2.0*gmstrr(i)/(pstar1(i)+hy_prght(i))
    gmstrl(i)=max(gmin(i),min(gmstrl(i),gmax(i)))
    gmstrr(i)=max(gmin(i),min(gmstrr(i),gmax(i)))
  

  end do

do i=5, numIntCells5, +1
  scrch1(i)=pstar1(i)-(gmstrl(i)-1.0)*hy_plft(i)/(hy_gmelft(i)-1.0)
  if (scrch1(i)==0) then 

    scrch1(i)=hy_smallp
  end if
  wlft1(i)=pstar1(i)+0.5*(gmstrl(i)-1.0)*(pstar1(i)+hy_plft(i))
  wlft1(i)=(pstar1(i)-hy_plft(i))*wlft1(i)/(hy_vlft(i)*scrch1(i))
  wlft1(i)=sqrt(abs(wlft1(i)))
  scrch2(i)=pstar1(i)-(gmstrr(i)-1.0)*hy_prght(i)/(hy_gmergt(i)-1.0)
  if (scrch2(i)==0.0) then 

    scrch2(i)=hy_smallp
  end if
  wrght1(i)=pstar1(i)+0.5*(gmstrr(i)-1.0)*(pstar1(i)+hy_prght(i))
  wrght1(i)=(pstar1(i)-hy_prght(i))*wrght1(i)/(hy_vrght(i)*scrch2(i))
  wrght1(i)=sqrt(abs(wrght1(i)))
  if (abs(pstar1(i)-hy_plft(i))<small_dp*(pstar1(i)+hy_plft(i))) then 

    wlft1(i)=hy_clft(i)
  end if
  wlft1(i)=max(wlft1(i),aux(i)*hy_clft(i))
  if (abs(pstar1(i)-hy_prght(i))<small_dp*((pstar1(i)+hy_prght(i)))) then 

    wrght1(i)=hy_crght(i)
  end if
  wrght1(i)=max(wrght1(i),aux(i)*hy_crght(i))


end do

  do i=5, numIntCells5-1, +2  
    pstar2(i)=hy_prght(i)-hy_plft(i)-wrght1(i)*(hy_urght(i)-hy_ulft(i))
    pstar2(i)=hy_plft(i)+pstar2(i)*wlft1(i)/(wlft1(i)+wrght1(i))
    pstar2(i)=max(hy_smallp,pstar2(i))
    pstar2((i+1))=hy_prght((i+1))-hy_plft((i+1))-wrght1((i+1))*(hy_urght((i+1))-hy_ulft((i+1)))
    pstar2((i+1))=hy_plft((i+1))+pstar2((i+1))*wlft1((i+1))/(wlft1((i+1))+wrght1((i+1)))
    pstar2((i+1))=max(hy_smallp,pstar2((i+1)))
  

  end do
  do i=(numIntCells5)-(MOD((numIntCells5), 2)), numIntCells5, +1  
    pstar2(i)=hy_prght(i)-hy_plft(i)-wrght1(i)*(hy_urght(i)-hy_ulft(i))
    pstar2(i)=hy_plft(i)+pstar2(i)*wlft1(i)/(wlft1(i)+wrght1(i))
    pstar2(i)=max(hy_smallp,pstar2(i))
  

  end do

do iii=5, numIntCells5, +2048
  do ii=iii, min(numIntCells5,iii+2032), +16  
    do i=ii, min(numIntCells5,ii+15)-2, +3    
      hy_pstor(1)=pstar1(i)
      hy_pstor(1)=pstar1((i+1))
      hy_pstor(1)=pstar1((i+2))
      hy_pstor(2)=pstar2(i)
      hy_pstor(2)=pstar2((i+1))
      hy_pstor(2)=pstar2((i+2))
      do nn=1, hy_nriem, +256      
        do n=nn, min(hy_nriem,nn+255)-9, +10        
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+2)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+3)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+4)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+5)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+6)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+7)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+8)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+9)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+10)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+11)=pstar(i)
          gmstrl((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_plft((i+1)))
          gmstrl((i+1))=hy_gmelft((i+1))+2*gmstrl((i+1))/(pstar2((i+1))+hy_plft((i+1)))
          gmstrr((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_prght((i+1)))
          gmstrr((i+1))=hy_gmergt((i+1))+2*gmstrr((i+1))/(pstar2((i+1))+hy_prght((i+1)))
          gmstrl((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrl((i+1))))
          gmstrr((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrr((i+1))))
          scrch1((i+1))=pstar2((i+1))-(gmstrl((i+1))-1)*hy_plft((i+1))/(hy_gmelft((i+1))-1)
          if (scrch1((i+1))==0.0) then 

            scrch1((i+1))=hy_smallp
          end if
          wlft((i+1))=pstar2((i+1))+0.5*(gmstrl((i+1))-1)*(pstar2((i+1))+hy_plft((i+1)))
          wlft((i+1))=(pstar2((i+1))-hy_plft((i+1)))*wlft((i+1))/(hy_vlft((i+1))*scrch1((i+1)))
          wlft((i+1))=sqrt(abs(wlft((i+1))))
          scrch2((i+1))=pstar2((i+1))-(gmstrr((i+1))-1)*hy_prght((i+1))/(hy_gmergt((i+1))-1)
          if (scrch2((i+1))==0.0) then 

            scrch2((i+1))=hy_smallp
          end if
          wrght((i+1))=pstar2((i+1))+0.5*(gmstrr((i+1))-1)*(pstar2((i+1))+hy_prght((i+1)))
          wrght((i+1))=(pstar2((i+1))-hy_prght((i+1)))*wrght((i+1))/(hy_vrght((i+1))*scrch2((i+1)))
          wrght((i+1))=sqrt(abs(wrght((i+1))))
          if (abs(pstar2((i+1))-hy_plft((i+1)))<small_dp*(pstar2((i+1))+hy_plft((i+1)))) then 

            wlft((i+1))=hy_clft((i+1))
          end if
          wlft((i+1))=max(wlft((i+1)),aux((i+1))*hy_clft((i+1)))
          if (abs(pstar2((i+1))-hy_prght((i+1)))<small_dp*(pstar2((i+1))+hy_prght((i+1)))) then 

            wrght((i+1))=hy_crght((i+1))
          end if
          wrght((i+1))=max(wrght((i+1)),aux((i+1))*hy_crght((i+1)))
          ustrl1=hy_ulft((i+1))-(pstar1((i+1))-hy_plft((i+1)))/wlft1((i+1))
          ustrr1=hy_urght((i+1))+(pstar1((i+1))-hy_prght((i+1)))/wrght1((i+1))
          ustrl2=hy_ulft((i+1))-(pstar2((i+1))-hy_plft((i+1)))/wlft((i+1))
          ustrr2=hy_urght((i+1))+(pstar2((i+1))-hy_prght((i+1)))/wrght((i+1))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+1))=delu2-delu1
          if (abs(pstar2((i+1))-pstar1((i+1)))<=hy_smallp) then 

            scrch1((i+1))=0.0
          end if
          if (abs(scrch1((i+1)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+1))=1.0
          
          end if
          pstar((i+1))=pstar2((i+1))-delu2*(pstar2((i+1))-pstar1((i+1)))/scrch1((i+1))
          pstar((i+1))=max(hy_smallp,pstar((i+1)))
          pres_err=abs(pstar((i+1))-pstar2((i+1)))/pstar((i+1))
          if (pres_err<hy_riemanTol) then 

            goto10((i+1))
          end if
          wlft1((i+1))=wlft((i+1))
          wrght1((i+1))=wrght((i+1))
          pstar1((i+1))=pstar2((i+1))
          pstar2((i+1))=pstar((i+1))
          hy_pstor(n+2)=pstar((i+1))
          gmstrl((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_plft((i+1)))
          gmstrl((i+1))=hy_gmelft((i+1))+2*gmstrl((i+1))/(pstar2((i+1))+hy_plft((i+1)))
          gmstrr((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_prght((i+1)))
          gmstrr((i+1))=hy_gmergt((i+1))+2*gmstrr((i+1))/(pstar2((i+1))+hy_prght((i+1)))
          gmstrl((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrl((i+1))))
          gmstrr((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrr((i+1))))
          scrch1((i+1))=pstar2((i+1))-(gmstrl((i+1))-1)*hy_plft((i+1))/(hy_gmelft((i+1))-1)
          if (scrch1((i+1))==0.0) then 

            scrch1((i+1))=hy_smallp
          end if
          wlft((i+1))=pstar2((i+1))+0.5*(gmstrl((i+1))-1)*(pstar2((i+1))+hy_plft((i+1)))
          wlft((i+1))=(pstar2((i+1))-hy_plft((i+1)))*wlft((i+1))/(hy_vlft((i+1))*scrch1((i+1)))
          wlft((i+1))=sqrt(abs(wlft((i+1))))
          scrch2((i+1))=pstar2((i+1))-(gmstrr((i+1))-1)*hy_prght((i+1))/(hy_gmergt((i+1))-1)
          if (scrch2((i+1))==0.0) then 

            scrch2((i+1))=hy_smallp
          end if
          wrght((i+1))=pstar2((i+1))+0.5*(gmstrr((i+1))-1)*(pstar2((i+1))+hy_prght((i+1)))
          wrght((i+1))=(pstar2((i+1))-hy_prght((i+1)))*wrght((i+1))/(hy_vrght((i+1))*scrch2((i+1)))
          wrght((i+1))=sqrt(abs(wrght((i+1))))
          if (abs(pstar2((i+1))-hy_plft((i+1)))<small_dp*(pstar2((i+1))+hy_plft((i+1)))) then 

            wlft((i+1))=hy_clft((i+1))
          end if
          wlft((i+1))=max(wlft((i+1)),aux((i+1))*hy_clft((i+1)))
          if (abs(pstar2((i+1))-hy_prght((i+1)))<small_dp*(pstar2((i+1))+hy_prght((i+1)))) then 

            wrght((i+1))=hy_crght((i+1))
          end if
          wrght((i+1))=max(wrght((i+1)),aux((i+1))*hy_crght((i+1)))
          ustrl1=hy_ulft((i+1))-(pstar1((i+1))-hy_plft((i+1)))/wlft1((i+1))
          ustrr1=hy_urght((i+1))+(pstar1((i+1))-hy_prght((i+1)))/wrght1((i+1))
          ustrl2=hy_ulft((i+1))-(pstar2((i+1))-hy_plft((i+1)))/wlft((i+1))
          ustrr2=hy_urght((i+1))+(pstar2((i+1))-hy_prght((i+1)))/wrght((i+1))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+1))=delu2-delu1
          if (abs(pstar2((i+1))-pstar1((i+1)))<=hy_smallp) then 

            scrch1((i+1))=0.0
          end if
          if (abs(scrch1((i+1)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+1))=1.0
          
          end if
          pstar((i+1))=pstar2((i+1))-delu2*(pstar2((i+1))-pstar1((i+1)))/scrch1((i+1))
          pstar((i+1))=max(hy_smallp,pstar((i+1)))
          pres_err=abs(pstar((i+1))-pstar2((i+1)))/pstar((i+1))
          if (pres_err<hy_riemanTol) then 

            goto10((i+1))
          end if
          wlft1((i+1))=wlft((i+1))
          wrght1((i+1))=wrght((i+1))
          pstar1((i+1))=pstar2((i+1))
          pstar2((i+1))=pstar((i+1))
          hy_pstor(n+3)=pstar((i+1))
          gmstrl((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_plft((i+1)))
          gmstrl((i+1))=hy_gmelft((i+1))+2*gmstrl((i+1))/(pstar2((i+1))+hy_plft((i+1)))
          gmstrr((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_prght((i+1)))
          gmstrr((i+1))=hy_gmergt((i+1))+2*gmstrr((i+1))/(pstar2((i+1))+hy_prght((i+1)))
          gmstrl((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrl((i+1))))
          gmstrr((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrr((i+1))))
          scrch1((i+1))=pstar2((i+1))-(gmstrl((i+1))-1)*hy_plft((i+1))/(hy_gmelft((i+1))-1)
          if (scrch1((i+1))==0.0) then 

            scrch1((i+1))=hy_smallp
          end if
          wlft((i+1))=pstar2((i+1))+0.5*(gmstrl((i+1))-1)*(pstar2((i+1))+hy_plft((i+1)))
          wlft((i+1))=(pstar2((i+1))-hy_plft((i+1)))*wlft((i+1))/(hy_vlft((i+1))*scrch1((i+1)))
          wlft((i+1))=sqrt(abs(wlft((i+1))))
          scrch2((i+1))=pstar2((i+1))-(gmstrr((i+1))-1)*hy_prght((i+1))/(hy_gmergt((i+1))-1)
          if (scrch2((i+1))==0.0) then 

            scrch2((i+1))=hy_smallp
          end if
          wrght((i+1))=pstar2((i+1))+0.5*(gmstrr((i+1))-1)*(pstar2((i+1))+hy_prght((i+1)))
          wrght((i+1))=(pstar2((i+1))-hy_prght((i+1)))*wrght((i+1))/(hy_vrght((i+1))*scrch2((i+1)))
          wrght((i+1))=sqrt(abs(wrght((i+1))))
          if (abs(pstar2((i+1))-hy_plft((i+1)))<small_dp*(pstar2((i+1))+hy_plft((i+1)))) then 

            wlft((i+1))=hy_clft((i+1))
          end if
          wlft((i+1))=max(wlft((i+1)),aux((i+1))*hy_clft((i+1)))
          if (abs(pstar2((i+1))-hy_prght((i+1)))<small_dp*(pstar2((i+1))+hy_prght((i+1)))) then 

            wrght((i+1))=hy_crght((i+1))
          end if
          wrght((i+1))=max(wrght((i+1)),aux((i+1))*hy_crght((i+1)))
          ustrl1=hy_ulft((i+1))-(pstar1((i+1))-hy_plft((i+1)))/wlft1((i+1))
          ustrr1=hy_urght((i+1))+(pstar1((i+1))-hy_prght((i+1)))/wrght1((i+1))
          ustrl2=hy_ulft((i+1))-(pstar2((i+1))-hy_plft((i+1)))/wlft((i+1))
          ustrr2=hy_urght((i+1))+(pstar2((i+1))-hy_prght((i+1)))/wrght((i+1))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+1))=delu2-delu1
          if (abs(pstar2((i+1))-pstar1((i+1)))<=hy_smallp) then 

            scrch1((i+1))=0.0
          end if
          if (abs(scrch1((i+1)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+1))=1.0
          
          end if
          pstar((i+1))=pstar2((i+1))-delu2*(pstar2((i+1))-pstar1((i+1)))/scrch1((i+1))
          pstar((i+1))=max(hy_smallp,pstar((i+1)))
          pres_err=abs(pstar((i+1))-pstar2((i+1)))/pstar((i+1))
          if (pres_err<hy_riemanTol) then 

            goto10((i+1))
          end if
          wlft1((i+1))=wlft((i+1))
          wrght1((i+1))=wrght((i+1))
          pstar1((i+1))=pstar2((i+1))
          pstar2((i+1))=pstar((i+1))
          hy_pstor(n+4)=pstar((i+1))
          gmstrl((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_plft((i+1)))
          gmstrl((i+1))=hy_gmelft((i+1))+2*gmstrl((i+1))/(pstar2((i+1))+hy_plft((i+1)))
          gmstrr((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_prght((i+1)))
          gmstrr((i+1))=hy_gmergt((i+1))+2*gmstrr((i+1))/(pstar2((i+1))+hy_prght((i+1)))
          gmstrl((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrl((i+1))))
          gmstrr((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrr((i+1))))
          scrch1((i+1))=pstar2((i+1))-(gmstrl((i+1))-1)*hy_plft((i+1))/(hy_gmelft((i+1))-1)
          if (scrch1((i+1))==0.0) then 

            scrch1((i+1))=hy_smallp
          end if
          wlft((i+1))=pstar2((i+1))+0.5*(gmstrl((i+1))-1)*(pstar2((i+1))+hy_plft((i+1)))
          wlft((i+1))=(pstar2((i+1))-hy_plft((i+1)))*wlft((i+1))/(hy_vlft((i+1))*scrch1((i+1)))
          wlft((i+1))=sqrt(abs(wlft((i+1))))
          scrch2((i+1))=pstar2((i+1))-(gmstrr((i+1))-1)*hy_prght((i+1))/(hy_gmergt((i+1))-1)
          if (scrch2((i+1))==0.0) then 

            scrch2((i+1))=hy_smallp
          end if
          wrght((i+1))=pstar2((i+1))+0.5*(gmstrr((i+1))-1)*(pstar2((i+1))+hy_prght((i+1)))
          wrght((i+1))=(pstar2((i+1))-hy_prght((i+1)))*wrght((i+1))/(hy_vrght((i+1))*scrch2((i+1)))
          wrght((i+1))=sqrt(abs(wrght((i+1))))
          if (abs(pstar2((i+1))-hy_plft((i+1)))<small_dp*(pstar2((i+1))+hy_plft((i+1)))) then 

            wlft((i+1))=hy_clft((i+1))
          end if
          wlft((i+1))=max(wlft((i+1)),aux((i+1))*hy_clft((i+1)))
          if (abs(pstar2((i+1))-hy_prght((i+1)))<small_dp*(pstar2((i+1))+hy_prght((i+1)))) then 

            wrght((i+1))=hy_crght((i+1))
          end if
          wrght((i+1))=max(wrght((i+1)),aux((i+1))*hy_crght((i+1)))
          ustrl1=hy_ulft((i+1))-(pstar1((i+1))-hy_plft((i+1)))/wlft1((i+1))
          ustrr1=hy_urght((i+1))+(pstar1((i+1))-hy_prght((i+1)))/wrght1((i+1))
          ustrl2=hy_ulft((i+1))-(pstar2((i+1))-hy_plft((i+1)))/wlft((i+1))
          ustrr2=hy_urght((i+1))+(pstar2((i+1))-hy_prght((i+1)))/wrght((i+1))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+1))=delu2-delu1
          if (abs(pstar2((i+1))-pstar1((i+1)))<=hy_smallp) then 

            scrch1((i+1))=0.0
          end if
          if (abs(scrch1((i+1)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+1))=1.0
          
          end if
          pstar((i+1))=pstar2((i+1))-delu2*(pstar2((i+1))-pstar1((i+1)))/scrch1((i+1))
          pstar((i+1))=max(hy_smallp,pstar((i+1)))
          pres_err=abs(pstar((i+1))-pstar2((i+1)))/pstar((i+1))
          if (pres_err<hy_riemanTol) then 

            goto10((i+1))
          end if
          wlft1((i+1))=wlft((i+1))
          wrght1((i+1))=wrght((i+1))
          pstar1((i+1))=pstar2((i+1))
          pstar2((i+1))=pstar((i+1))
          hy_pstor(n+5)=pstar((i+1))
          gmstrl((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_plft((i+1)))
          gmstrl((i+1))=hy_gmelft((i+1))+2*gmstrl((i+1))/(pstar2((i+1))+hy_plft((i+1)))
          gmstrr((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_prght((i+1)))
          gmstrr((i+1))=hy_gmergt((i+1))+2*gmstrr((i+1))/(pstar2((i+1))+hy_prght((i+1)))
          gmstrl((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrl((i+1))))
          gmstrr((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrr((i+1))))
          scrch1((i+1))=pstar2((i+1))-(gmstrl((i+1))-1)*hy_plft((i+1))/(hy_gmelft((i+1))-1)
          if (scrch1((i+1))==0.0) then 

            scrch1((i+1))=hy_smallp
          end if
          wlft((i+1))=pstar2((i+1))+0.5*(gmstrl((i+1))-1)*(pstar2((i+1))+hy_plft((i+1)))
          wlft((i+1))=(pstar2((i+1))-hy_plft((i+1)))*wlft((i+1))/(hy_vlft((i+1))*scrch1((i+1)))
          wlft((i+1))=sqrt(abs(wlft((i+1))))
          scrch2((i+1))=pstar2((i+1))-(gmstrr((i+1))-1)*hy_prght((i+1))/(hy_gmergt((i+1))-1)
          if (scrch2((i+1))==0.0) then 

            scrch2((i+1))=hy_smallp
          end if
          wrght((i+1))=pstar2((i+1))+0.5*(gmstrr((i+1))-1)*(pstar2((i+1))+hy_prght((i+1)))
          wrght((i+1))=(pstar2((i+1))-hy_prght((i+1)))*wrght((i+1))/(hy_vrght((i+1))*scrch2((i+1)))
          wrght((i+1))=sqrt(abs(wrght((i+1))))
          if (abs(pstar2((i+1))-hy_plft((i+1)))<small_dp*(pstar2((i+1))+hy_plft((i+1)))) then 

            wlft((i+1))=hy_clft((i+1))
          end if
          wlft((i+1))=max(wlft((i+1)),aux((i+1))*hy_clft((i+1)))
          if (abs(pstar2((i+1))-hy_prght((i+1)))<small_dp*(pstar2((i+1))+hy_prght((i+1)))) then 

            wrght((i+1))=hy_crght((i+1))
          end if
          wrght((i+1))=max(wrght((i+1)),aux((i+1))*hy_crght((i+1)))
          ustrl1=hy_ulft((i+1))-(pstar1((i+1))-hy_plft((i+1)))/wlft1((i+1))
          ustrr1=hy_urght((i+1))+(pstar1((i+1))-hy_prght((i+1)))/wrght1((i+1))
          ustrl2=hy_ulft((i+1))-(pstar2((i+1))-hy_plft((i+1)))/wlft((i+1))
          ustrr2=hy_urght((i+1))+(pstar2((i+1))-hy_prght((i+1)))/wrght((i+1))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+1))=delu2-delu1
          if (abs(pstar2((i+1))-pstar1((i+1)))<=hy_smallp) then 

            scrch1((i+1))=0.0
          end if
          if (abs(scrch1((i+1)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+1))=1.0
          
          end if
          pstar((i+1))=pstar2((i+1))-delu2*(pstar2((i+1))-pstar1((i+1)))/scrch1((i+1))
          pstar((i+1))=max(hy_smallp,pstar((i+1)))
          pres_err=abs(pstar((i+1))-pstar2((i+1)))/pstar((i+1))
          if (pres_err<hy_riemanTol) then 

            goto10((i+1))
          end if
          wlft1((i+1))=wlft((i+1))
          wrght1((i+1))=wrght((i+1))
          pstar1((i+1))=pstar2((i+1))
          pstar2((i+1))=pstar((i+1))
          hy_pstor(n+6)=pstar((i+1))
          gmstrl((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_plft((i+1)))
          gmstrl((i+1))=hy_gmelft((i+1))+2*gmstrl((i+1))/(pstar2((i+1))+hy_plft((i+1)))
          gmstrr((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_prght((i+1)))
          gmstrr((i+1))=hy_gmergt((i+1))+2*gmstrr((i+1))/(pstar2((i+1))+hy_prght((i+1)))
          gmstrl((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrl((i+1))))
          gmstrr((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrr((i+1))))
          scrch1((i+1))=pstar2((i+1))-(gmstrl((i+1))-1)*hy_plft((i+1))/(hy_gmelft((i+1))-1)
          if (scrch1((i+1))==0.0) then 

            scrch1((i+1))=hy_smallp
          end if
          wlft((i+1))=pstar2((i+1))+0.5*(gmstrl((i+1))-1)*(pstar2((i+1))+hy_plft((i+1)))
          wlft((i+1))=(pstar2((i+1))-hy_plft((i+1)))*wlft((i+1))/(hy_vlft((i+1))*scrch1((i+1)))
          wlft((i+1))=sqrt(abs(wlft((i+1))))
          scrch2((i+1))=pstar2((i+1))-(gmstrr((i+1))-1)*hy_prght((i+1))/(hy_gmergt((i+1))-1)
          if (scrch2((i+1))==0.0) then 

            scrch2((i+1))=hy_smallp
          end if
          wrght((i+1))=pstar2((i+1))+0.5*(gmstrr((i+1))-1)*(pstar2((i+1))+hy_prght((i+1)))
          wrght((i+1))=(pstar2((i+1))-hy_prght((i+1)))*wrght((i+1))/(hy_vrght((i+1))*scrch2((i+1)))
          wrght((i+1))=sqrt(abs(wrght((i+1))))
          if (abs(pstar2((i+1))-hy_plft((i+1)))<small_dp*(pstar2((i+1))+hy_plft((i+1)))) then 

            wlft((i+1))=hy_clft((i+1))
          end if
          wlft((i+1))=max(wlft((i+1)),aux((i+1))*hy_clft((i+1)))
          if (abs(pstar2((i+1))-hy_prght((i+1)))<small_dp*(pstar2((i+1))+hy_prght((i+1)))) then 

            wrght((i+1))=hy_crght((i+1))
          end if
          wrght((i+1))=max(wrght((i+1)),aux((i+1))*hy_crght((i+1)))
          ustrl1=hy_ulft((i+1))-(pstar1((i+1))-hy_plft((i+1)))/wlft1((i+1))
          ustrr1=hy_urght((i+1))+(pstar1((i+1))-hy_prght((i+1)))/wrght1((i+1))
          ustrl2=hy_ulft((i+1))-(pstar2((i+1))-hy_plft((i+1)))/wlft((i+1))
          ustrr2=hy_urght((i+1))+(pstar2((i+1))-hy_prght((i+1)))/wrght((i+1))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+1))=delu2-delu1
          if (abs(pstar2((i+1))-pstar1((i+1)))<=hy_smallp) then 

            scrch1((i+1))=0.0
          end if
          if (abs(scrch1((i+1)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+1))=1.0
          
          end if
          pstar((i+1))=pstar2((i+1))-delu2*(pstar2((i+1))-pstar1((i+1)))/scrch1((i+1))
          pstar((i+1))=max(hy_smallp,pstar((i+1)))
          pres_err=abs(pstar((i+1))-pstar2((i+1)))/pstar((i+1))
          if (pres_err<hy_riemanTol) then 

            goto10((i+1))
          end if
          wlft1((i+1))=wlft((i+1))
          wrght1((i+1))=wrght((i+1))
          pstar1((i+1))=pstar2((i+1))
          pstar2((i+1))=pstar((i+1))
          hy_pstor(n+7)=pstar((i+1))
          gmstrl((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_plft((i+1)))
          gmstrl((i+1))=hy_gmelft((i+1))+2*gmstrl((i+1))/(pstar2((i+1))+hy_plft((i+1)))
          gmstrr((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_prght((i+1)))
          gmstrr((i+1))=hy_gmergt((i+1))+2*gmstrr((i+1))/(pstar2((i+1))+hy_prght((i+1)))
          gmstrl((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrl((i+1))))
          gmstrr((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrr((i+1))))
          scrch1((i+1))=pstar2((i+1))-(gmstrl((i+1))-1)*hy_plft((i+1))/(hy_gmelft((i+1))-1)
          if (scrch1((i+1))==0.0) then 

            scrch1((i+1))=hy_smallp
          end if
          wlft((i+1))=pstar2((i+1))+0.5*(gmstrl((i+1))-1)*(pstar2((i+1))+hy_plft((i+1)))
          wlft((i+1))=(pstar2((i+1))-hy_plft((i+1)))*wlft((i+1))/(hy_vlft((i+1))*scrch1((i+1)))
          wlft((i+1))=sqrt(abs(wlft((i+1))))
          scrch2((i+1))=pstar2((i+1))-(gmstrr((i+1))-1)*hy_prght((i+1))/(hy_gmergt((i+1))-1)
          if (scrch2((i+1))==0.0) then 

            scrch2((i+1))=hy_smallp
          end if
          wrght((i+1))=pstar2((i+1))+0.5*(gmstrr((i+1))-1)*(pstar2((i+1))+hy_prght((i+1)))
          wrght((i+1))=(pstar2((i+1))-hy_prght((i+1)))*wrght((i+1))/(hy_vrght((i+1))*scrch2((i+1)))
          wrght((i+1))=sqrt(abs(wrght((i+1))))
          if (abs(pstar2((i+1))-hy_plft((i+1)))<small_dp*(pstar2((i+1))+hy_plft((i+1)))) then 

            wlft((i+1))=hy_clft((i+1))
          end if
          wlft((i+1))=max(wlft((i+1)),aux((i+1))*hy_clft((i+1)))
          if (abs(pstar2((i+1))-hy_prght((i+1)))<small_dp*(pstar2((i+1))+hy_prght((i+1)))) then 

            wrght((i+1))=hy_crght((i+1))
          end if
          wrght((i+1))=max(wrght((i+1)),aux((i+1))*hy_crght((i+1)))
          ustrl1=hy_ulft((i+1))-(pstar1((i+1))-hy_plft((i+1)))/wlft1((i+1))
          ustrr1=hy_urght((i+1))+(pstar1((i+1))-hy_prght((i+1)))/wrght1((i+1))
          ustrl2=hy_ulft((i+1))-(pstar2((i+1))-hy_plft((i+1)))/wlft((i+1))
          ustrr2=hy_urght((i+1))+(pstar2((i+1))-hy_prght((i+1)))/wrght((i+1))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+1))=delu2-delu1
          if (abs(pstar2((i+1))-pstar1((i+1)))<=hy_smallp) then 

            scrch1((i+1))=0.0
          end if
          if (abs(scrch1((i+1)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+1))=1.0
          
          end if
          pstar((i+1))=pstar2((i+1))-delu2*(pstar2((i+1))-pstar1((i+1)))/scrch1((i+1))
          pstar((i+1))=max(hy_smallp,pstar((i+1)))
          pres_err=abs(pstar((i+1))-pstar2((i+1)))/pstar((i+1))
          if (pres_err<hy_riemanTol) then 

            goto10((i+1))
          end if
          wlft1((i+1))=wlft((i+1))
          wrght1((i+1))=wrght((i+1))
          pstar1((i+1))=pstar2((i+1))
          pstar2((i+1))=pstar((i+1))
          hy_pstor(n+8)=pstar((i+1))
          gmstrl((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_plft((i+1)))
          gmstrl((i+1))=hy_gmelft((i+1))+2*gmstrl((i+1))/(pstar2((i+1))+hy_plft((i+1)))
          gmstrr((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_prght((i+1)))
          gmstrr((i+1))=hy_gmergt((i+1))+2*gmstrr((i+1))/(pstar2((i+1))+hy_prght((i+1)))
          gmstrl((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrl((i+1))))
          gmstrr((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrr((i+1))))
          scrch1((i+1))=pstar2((i+1))-(gmstrl((i+1))-1)*hy_plft((i+1))/(hy_gmelft((i+1))-1)
          if (scrch1((i+1))==0.0) then 

            scrch1((i+1))=hy_smallp
          end if
          wlft((i+1))=pstar2((i+1))+0.5*(gmstrl((i+1))-1)*(pstar2((i+1))+hy_plft((i+1)))
          wlft((i+1))=(pstar2((i+1))-hy_plft((i+1)))*wlft((i+1))/(hy_vlft((i+1))*scrch1((i+1)))
          wlft((i+1))=sqrt(abs(wlft((i+1))))
          scrch2((i+1))=pstar2((i+1))-(gmstrr((i+1))-1)*hy_prght((i+1))/(hy_gmergt((i+1))-1)
          if (scrch2((i+1))==0.0) then 

            scrch2((i+1))=hy_smallp
          end if
          wrght((i+1))=pstar2((i+1))+0.5*(gmstrr((i+1))-1)*(pstar2((i+1))+hy_prght((i+1)))
          wrght((i+1))=(pstar2((i+1))-hy_prght((i+1)))*wrght((i+1))/(hy_vrght((i+1))*scrch2((i+1)))
          wrght((i+1))=sqrt(abs(wrght((i+1))))
          if (abs(pstar2((i+1))-hy_plft((i+1)))<small_dp*(pstar2((i+1))+hy_plft((i+1)))) then 

            wlft((i+1))=hy_clft((i+1))
          end if
          wlft((i+1))=max(wlft((i+1)),aux((i+1))*hy_clft((i+1)))
          if (abs(pstar2((i+1))-hy_prght((i+1)))<small_dp*(pstar2((i+1))+hy_prght((i+1)))) then 

            wrght((i+1))=hy_crght((i+1))
          end if
          wrght((i+1))=max(wrght((i+1)),aux((i+1))*hy_crght((i+1)))
          ustrl1=hy_ulft((i+1))-(pstar1((i+1))-hy_plft((i+1)))/wlft1((i+1))
          ustrr1=hy_urght((i+1))+(pstar1((i+1))-hy_prght((i+1)))/wrght1((i+1))
          ustrl2=hy_ulft((i+1))-(pstar2((i+1))-hy_plft((i+1)))/wlft((i+1))
          ustrr2=hy_urght((i+1))+(pstar2((i+1))-hy_prght((i+1)))/wrght((i+1))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+1))=delu2-delu1
          if (abs(pstar2((i+1))-pstar1((i+1)))<=hy_smallp) then 

            scrch1((i+1))=0.0
          end if
          if (abs(scrch1((i+1)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+1))=1.0
          
          end if
          pstar((i+1))=pstar2((i+1))-delu2*(pstar2((i+1))-pstar1((i+1)))/scrch1((i+1))
          pstar((i+1))=max(hy_smallp,pstar((i+1)))
          pres_err=abs(pstar((i+1))-pstar2((i+1)))/pstar((i+1))
          if (pres_err<hy_riemanTol) then 

            goto10((i+1))
          end if
          wlft1((i+1))=wlft((i+1))
          wrght1((i+1))=wrght((i+1))
          pstar1((i+1))=pstar2((i+1))
          pstar2((i+1))=pstar((i+1))
          hy_pstor(n+9)=pstar((i+1))
          gmstrl((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_plft((i+1)))
          gmstrl((i+1))=hy_gmelft((i+1))+2*gmstrl((i+1))/(pstar2((i+1))+hy_plft((i+1)))
          gmstrr((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_prght((i+1)))
          gmstrr((i+1))=hy_gmergt((i+1))+2*gmstrr((i+1))/(pstar2((i+1))+hy_prght((i+1)))
          gmstrl((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrl((i+1))))
          gmstrr((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrr((i+1))))
          scrch1((i+1))=pstar2((i+1))-(gmstrl((i+1))-1)*hy_plft((i+1))/(hy_gmelft((i+1))-1)
          if (scrch1((i+1))==0.0) then 

            scrch1((i+1))=hy_smallp
          end if
          wlft((i+1))=pstar2((i+1))+0.5*(gmstrl((i+1))-1)*(pstar2((i+1))+hy_plft((i+1)))
          wlft((i+1))=(pstar2((i+1))-hy_plft((i+1)))*wlft((i+1))/(hy_vlft((i+1))*scrch1((i+1)))
          wlft((i+1))=sqrt(abs(wlft((i+1))))
          scrch2((i+1))=pstar2((i+1))-(gmstrr((i+1))-1)*hy_prght((i+1))/(hy_gmergt((i+1))-1)
          if (scrch2((i+1))==0.0) then 

            scrch2((i+1))=hy_smallp
          end if
          wrght((i+1))=pstar2((i+1))+0.5*(gmstrr((i+1))-1)*(pstar2((i+1))+hy_prght((i+1)))
          wrght((i+1))=(pstar2((i+1))-hy_prght((i+1)))*wrght((i+1))/(hy_vrght((i+1))*scrch2((i+1)))
          wrght((i+1))=sqrt(abs(wrght((i+1))))
          if (abs(pstar2((i+1))-hy_plft((i+1)))<small_dp*(pstar2((i+1))+hy_plft((i+1)))) then 

            wlft((i+1))=hy_clft((i+1))
          end if
          wlft((i+1))=max(wlft((i+1)),aux((i+1))*hy_clft((i+1)))
          if (abs(pstar2((i+1))-hy_prght((i+1)))<small_dp*(pstar2((i+1))+hy_prght((i+1)))) then 

            wrght((i+1))=hy_crght((i+1))
          end if
          wrght((i+1))=max(wrght((i+1)),aux((i+1))*hy_crght((i+1)))
          ustrl1=hy_ulft((i+1))-(pstar1((i+1))-hy_plft((i+1)))/wlft1((i+1))
          ustrr1=hy_urght((i+1))+(pstar1((i+1))-hy_prght((i+1)))/wrght1((i+1))
          ustrl2=hy_ulft((i+1))-(pstar2((i+1))-hy_plft((i+1)))/wlft((i+1))
          ustrr2=hy_urght((i+1))+(pstar2((i+1))-hy_prght((i+1)))/wrght((i+1))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+1))=delu2-delu1
          if (abs(pstar2((i+1))-pstar1((i+1)))<=hy_smallp) then 

            scrch1((i+1))=0.0
          end if
          if (abs(scrch1((i+1)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+1))=1.0
          
          end if
          pstar((i+1))=pstar2((i+1))-delu2*(pstar2((i+1))-pstar1((i+1)))/scrch1((i+1))
          pstar((i+1))=max(hy_smallp,pstar((i+1)))
          pres_err=abs(pstar((i+1))-pstar2((i+1)))/pstar((i+1))
          if (pres_err<hy_riemanTol) then 

            goto10((i+1))
          end if
          wlft1((i+1))=wlft((i+1))
          wrght1((i+1))=wrght((i+1))
          pstar1((i+1))=pstar2((i+1))
          pstar2((i+1))=pstar((i+1))
          hy_pstor(n+10)=pstar((i+1))
          gmstrl((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_plft((i+1)))
          gmstrl((i+1))=hy_gmelft((i+1))+2*gmstrl((i+1))/(pstar2((i+1))+hy_plft((i+1)))
          gmstrr((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_prght((i+1)))
          gmstrr((i+1))=hy_gmergt((i+1))+2*gmstrr((i+1))/(pstar2((i+1))+hy_prght((i+1)))
          gmstrl((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrl((i+1))))
          gmstrr((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrr((i+1))))
          scrch1((i+1))=pstar2((i+1))-(gmstrl((i+1))-1)*hy_plft((i+1))/(hy_gmelft((i+1))-1)
          if (scrch1((i+1))==0.0) then 

            scrch1((i+1))=hy_smallp
          end if
          wlft((i+1))=pstar2((i+1))+0.5*(gmstrl((i+1))-1)*(pstar2((i+1))+hy_plft((i+1)))
          wlft((i+1))=(pstar2((i+1))-hy_plft((i+1)))*wlft((i+1))/(hy_vlft((i+1))*scrch1((i+1)))
          wlft((i+1))=sqrt(abs(wlft((i+1))))
          scrch2((i+1))=pstar2((i+1))-(gmstrr((i+1))-1)*hy_prght((i+1))/(hy_gmergt((i+1))-1)
          if (scrch2((i+1))==0.0) then 

            scrch2((i+1))=hy_smallp
          end if
          wrght((i+1))=pstar2((i+1))+0.5*(gmstrr((i+1))-1)*(pstar2((i+1))+hy_prght((i+1)))
          wrght((i+1))=(pstar2((i+1))-hy_prght((i+1)))*wrght((i+1))/(hy_vrght((i+1))*scrch2((i+1)))
          wrght((i+1))=sqrt(abs(wrght((i+1))))
          if (abs(pstar2((i+1))-hy_plft((i+1)))<small_dp*(pstar2((i+1))+hy_plft((i+1)))) then 

            wlft((i+1))=hy_clft((i+1))
          end if
          wlft((i+1))=max(wlft((i+1)),aux((i+1))*hy_clft((i+1)))
          if (abs(pstar2((i+1))-hy_prght((i+1)))<small_dp*(pstar2((i+1))+hy_prght((i+1)))) then 

            wrght((i+1))=hy_crght((i+1))
          end if
          wrght((i+1))=max(wrght((i+1)),aux((i+1))*hy_crght((i+1)))
          ustrl1=hy_ulft((i+1))-(pstar1((i+1))-hy_plft((i+1)))/wlft1((i+1))
          ustrr1=hy_urght((i+1))+(pstar1((i+1))-hy_prght((i+1)))/wrght1((i+1))
          ustrl2=hy_ulft((i+1))-(pstar2((i+1))-hy_plft((i+1)))/wlft((i+1))
          ustrr2=hy_urght((i+1))+(pstar2((i+1))-hy_prght((i+1)))/wrght((i+1))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+1))=delu2-delu1
          if (abs(pstar2((i+1))-pstar1((i+1)))<=hy_smallp) then 

            scrch1((i+1))=0.0
          end if
          if (abs(scrch1((i+1)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+1))=1.0
          
          end if
          pstar((i+1))=pstar2((i+1))-delu2*(pstar2((i+1))-pstar1((i+1)))/scrch1((i+1))
          pstar((i+1))=max(hy_smallp,pstar((i+1)))
          pres_err=abs(pstar((i+1))-pstar2((i+1)))/pstar((i+1))
          if (pres_err<hy_riemanTol) then 

            goto10((i+1))
          end if
          wlft1((i+1))=wlft((i+1))
          wrght1((i+1))=wrght((i+1))
          pstar1((i+1))=pstar2((i+1))
          pstar2((i+1))=pstar((i+1))
          hy_pstor(n+11)=pstar((i+1))
          gmstrl((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_plft((i+2)))
          gmstrl((i+2))=hy_gmelft((i+2))+2*gmstrl((i+2))/(pstar2((i+2))+hy_plft((i+2)))
          gmstrr((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_prght((i+2)))
          gmstrr((i+2))=hy_gmergt((i+2))+2*gmstrr((i+2))/(pstar2((i+2))+hy_prght((i+2)))
          gmstrl((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrl((i+2))))
          gmstrr((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrr((i+2))))
          scrch1((i+2))=pstar2((i+2))-(gmstrl((i+2))-1)*hy_plft((i+2))/(hy_gmelft((i+2))-1)
          if (scrch1((i+2))==0.0) then 

            scrch1((i+2))=hy_smallp
          end if
          wlft((i+2))=pstar2((i+2))+0.5*(gmstrl((i+2))-1)*(pstar2((i+2))+hy_plft((i+2)))
          wlft((i+2))=(pstar2((i+2))-hy_plft((i+2)))*wlft((i+2))/(hy_vlft((i+2))*scrch1((i+2)))
          wlft((i+2))=sqrt(abs(wlft((i+2))))
          scrch2((i+2))=pstar2((i+2))-(gmstrr((i+2))-1)*hy_prght((i+2))/(hy_gmergt((i+2))-1)
          if (scrch2((i+2))==0.0) then 

            scrch2((i+2))=hy_smallp
          end if
          wrght((i+2))=pstar2((i+2))+0.5*(gmstrr((i+2))-1)*(pstar2((i+2))+hy_prght((i+2)))
          wrght((i+2))=(pstar2((i+2))-hy_prght((i+2)))*wrght((i+2))/(hy_vrght((i+2))*scrch2((i+2)))
          wrght((i+2))=sqrt(abs(wrght((i+2))))
          if (abs(pstar2((i+2))-hy_plft((i+2)))<small_dp*(pstar2((i+2))+hy_plft((i+2)))) then 

            wlft((i+2))=hy_clft((i+2))
          end if
          wlft((i+2))=max(wlft((i+2)),aux((i+2))*hy_clft((i+2)))
          if (abs(pstar2((i+2))-hy_prght((i+2)))<small_dp*(pstar2((i+2))+hy_prght((i+2)))) then 

            wrght((i+2))=hy_crght((i+2))
          end if
          wrght((i+2))=max(wrght((i+2)),aux((i+2))*hy_crght((i+2)))
          ustrl1=hy_ulft((i+2))-(pstar1((i+2))-hy_plft((i+2)))/wlft1((i+2))
          ustrr1=hy_urght((i+2))+(pstar1((i+2))-hy_prght((i+2)))/wrght1((i+2))
          ustrl2=hy_ulft((i+2))-(pstar2((i+2))-hy_plft((i+2)))/wlft((i+2))
          ustrr2=hy_urght((i+2))+(pstar2((i+2))-hy_prght((i+2)))/wrght((i+2))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+2))=delu2-delu1
          if (abs(pstar2((i+2))-pstar1((i+2)))<=hy_smallp) then 

            scrch1((i+2))=0.0
          end if
          if (abs(scrch1((i+2)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+2))=1.0
          
          end if
          pstar((i+2))=pstar2((i+2))-delu2*(pstar2((i+2))-pstar1((i+2)))/scrch1((i+2))
          pstar((i+2))=max(hy_smallp,pstar((i+2)))
          pres_err=abs(pstar((i+2))-pstar2((i+2)))/pstar((i+2))
          if (pres_err<hy_riemanTol) then 

            goto10((i+2))
          end if
          wlft1((i+2))=wlft((i+2))
          wrght1((i+2))=wrght((i+2))
          pstar1((i+2))=pstar2((i+2))
          pstar2((i+2))=pstar((i+2))
          hy_pstor(n+2)=pstar((i+2))
          gmstrl((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_plft((i+2)))
          gmstrl((i+2))=hy_gmelft((i+2))+2*gmstrl((i+2))/(pstar2((i+2))+hy_plft((i+2)))
          gmstrr((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_prght((i+2)))
          gmstrr((i+2))=hy_gmergt((i+2))+2*gmstrr((i+2))/(pstar2((i+2))+hy_prght((i+2)))
          gmstrl((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrl((i+2))))
          gmstrr((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrr((i+2))))
          scrch1((i+2))=pstar2((i+2))-(gmstrl((i+2))-1)*hy_plft((i+2))/(hy_gmelft((i+2))-1)
          if (scrch1((i+2))==0.0) then 

            scrch1((i+2))=hy_smallp
          end if
          wlft((i+2))=pstar2((i+2))+0.5*(gmstrl((i+2))-1)*(pstar2((i+2))+hy_plft((i+2)))
          wlft((i+2))=(pstar2((i+2))-hy_plft((i+2)))*wlft((i+2))/(hy_vlft((i+2))*scrch1((i+2)))
          wlft((i+2))=sqrt(abs(wlft((i+2))))
          scrch2((i+2))=pstar2((i+2))-(gmstrr((i+2))-1)*hy_prght((i+2))/(hy_gmergt((i+2))-1)
          if (scrch2((i+2))==0.0) then 

            scrch2((i+2))=hy_smallp
          end if
          wrght((i+2))=pstar2((i+2))+0.5*(gmstrr((i+2))-1)*(pstar2((i+2))+hy_prght((i+2)))
          wrght((i+2))=(pstar2((i+2))-hy_prght((i+2)))*wrght((i+2))/(hy_vrght((i+2))*scrch2((i+2)))
          wrght((i+2))=sqrt(abs(wrght((i+2))))
          if (abs(pstar2((i+2))-hy_plft((i+2)))<small_dp*(pstar2((i+2))+hy_plft((i+2)))) then 

            wlft((i+2))=hy_clft((i+2))
          end if
          wlft((i+2))=max(wlft((i+2)),aux((i+2))*hy_clft((i+2)))
          if (abs(pstar2((i+2))-hy_prght((i+2)))<small_dp*(pstar2((i+2))+hy_prght((i+2)))) then 

            wrght((i+2))=hy_crght((i+2))
          end if
          wrght((i+2))=max(wrght((i+2)),aux((i+2))*hy_crght((i+2)))
          ustrl1=hy_ulft((i+2))-(pstar1((i+2))-hy_plft((i+2)))/wlft1((i+2))
          ustrr1=hy_urght((i+2))+(pstar1((i+2))-hy_prght((i+2)))/wrght1((i+2))
          ustrl2=hy_ulft((i+2))-(pstar2((i+2))-hy_plft((i+2)))/wlft((i+2))
          ustrr2=hy_urght((i+2))+(pstar2((i+2))-hy_prght((i+2)))/wrght((i+2))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+2))=delu2-delu1
          if (abs(pstar2((i+2))-pstar1((i+2)))<=hy_smallp) then 

            scrch1((i+2))=0.0
          end if
          if (abs(scrch1((i+2)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+2))=1.0
          
          end if
          pstar((i+2))=pstar2((i+2))-delu2*(pstar2((i+2))-pstar1((i+2)))/scrch1((i+2))
          pstar((i+2))=max(hy_smallp,pstar((i+2)))
          pres_err=abs(pstar((i+2))-pstar2((i+2)))/pstar((i+2))
          if (pres_err<hy_riemanTol) then 

            goto10((i+2))
          end if
          wlft1((i+2))=wlft((i+2))
          wrght1((i+2))=wrght((i+2))
          pstar1((i+2))=pstar2((i+2))
          pstar2((i+2))=pstar((i+2))
          hy_pstor(n+3)=pstar((i+2))
          gmstrl((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_plft((i+2)))
          gmstrl((i+2))=hy_gmelft((i+2))+2*gmstrl((i+2))/(pstar2((i+2))+hy_plft((i+2)))
          gmstrr((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_prght((i+2)))
          gmstrr((i+2))=hy_gmergt((i+2))+2*gmstrr((i+2))/(pstar2((i+2))+hy_prght((i+2)))
          gmstrl((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrl((i+2))))
          gmstrr((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrr((i+2))))
          scrch1((i+2))=pstar2((i+2))-(gmstrl((i+2))-1)*hy_plft((i+2))/(hy_gmelft((i+2))-1)
          if (scrch1((i+2))==0.0) then 

            scrch1((i+2))=hy_smallp
          end if
          wlft((i+2))=pstar2((i+2))+0.5*(gmstrl((i+2))-1)*(pstar2((i+2))+hy_plft((i+2)))
          wlft((i+2))=(pstar2((i+2))-hy_plft((i+2)))*wlft((i+2))/(hy_vlft((i+2))*scrch1((i+2)))
          wlft((i+2))=sqrt(abs(wlft((i+2))))
          scrch2((i+2))=pstar2((i+2))-(gmstrr((i+2))-1)*hy_prght((i+2))/(hy_gmergt((i+2))-1)
          if (scrch2((i+2))==0.0) then 

            scrch2((i+2))=hy_smallp
          end if
          wrght((i+2))=pstar2((i+2))+0.5*(gmstrr((i+2))-1)*(pstar2((i+2))+hy_prght((i+2)))
          wrght((i+2))=(pstar2((i+2))-hy_prght((i+2)))*wrght((i+2))/(hy_vrght((i+2))*scrch2((i+2)))
          wrght((i+2))=sqrt(abs(wrght((i+2))))
          if (abs(pstar2((i+2))-hy_plft((i+2)))<small_dp*(pstar2((i+2))+hy_plft((i+2)))) then 

            wlft((i+2))=hy_clft((i+2))
          end if
          wlft((i+2))=max(wlft((i+2)),aux((i+2))*hy_clft((i+2)))
          if (abs(pstar2((i+2))-hy_prght((i+2)))<small_dp*(pstar2((i+2))+hy_prght((i+2)))) then 

            wrght((i+2))=hy_crght((i+2))
          end if
          wrght((i+2))=max(wrght((i+2)),aux((i+2))*hy_crght((i+2)))
          ustrl1=hy_ulft((i+2))-(pstar1((i+2))-hy_plft((i+2)))/wlft1((i+2))
          ustrr1=hy_urght((i+2))+(pstar1((i+2))-hy_prght((i+2)))/wrght1((i+2))
          ustrl2=hy_ulft((i+2))-(pstar2((i+2))-hy_plft((i+2)))/wlft((i+2))
          ustrr2=hy_urght((i+2))+(pstar2((i+2))-hy_prght((i+2)))/wrght((i+2))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+2))=delu2-delu1
          if (abs(pstar2((i+2))-pstar1((i+2)))<=hy_smallp) then 

            scrch1((i+2))=0.0
          end if
          if (abs(scrch1((i+2)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+2))=1.0
          
          end if
          pstar((i+2))=pstar2((i+2))-delu2*(pstar2((i+2))-pstar1((i+2)))/scrch1((i+2))
          pstar((i+2))=max(hy_smallp,pstar((i+2)))
          pres_err=abs(pstar((i+2))-pstar2((i+2)))/pstar((i+2))
          if (pres_err<hy_riemanTol) then 

            goto10((i+2))
          end if
          wlft1((i+2))=wlft((i+2))
          wrght1((i+2))=wrght((i+2))
          pstar1((i+2))=pstar2((i+2))
          pstar2((i+2))=pstar((i+2))
          hy_pstor(n+4)=pstar((i+2))
          gmstrl((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_plft((i+2)))
          gmstrl((i+2))=hy_gmelft((i+2))+2*gmstrl((i+2))/(pstar2((i+2))+hy_plft((i+2)))
          gmstrr((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_prght((i+2)))
          gmstrr((i+2))=hy_gmergt((i+2))+2*gmstrr((i+2))/(pstar2((i+2))+hy_prght((i+2)))
          gmstrl((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrl((i+2))))
          gmstrr((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrr((i+2))))
          scrch1((i+2))=pstar2((i+2))-(gmstrl((i+2))-1)*hy_plft((i+2))/(hy_gmelft((i+2))-1)
          if (scrch1((i+2))==0.0) then 

            scrch1((i+2))=hy_smallp
          end if
          wlft((i+2))=pstar2((i+2))+0.5*(gmstrl((i+2))-1)*(pstar2((i+2))+hy_plft((i+2)))
          wlft((i+2))=(pstar2((i+2))-hy_plft((i+2)))*wlft((i+2))/(hy_vlft((i+2))*scrch1((i+2)))
          wlft((i+2))=sqrt(abs(wlft((i+2))))
          scrch2((i+2))=pstar2((i+2))-(gmstrr((i+2))-1)*hy_prght((i+2))/(hy_gmergt((i+2))-1)
          if (scrch2((i+2))==0.0) then 

            scrch2((i+2))=hy_smallp
          end if
          wrght((i+2))=pstar2((i+2))+0.5*(gmstrr((i+2))-1)*(pstar2((i+2))+hy_prght((i+2)))
          wrght((i+2))=(pstar2((i+2))-hy_prght((i+2)))*wrght((i+2))/(hy_vrght((i+2))*scrch2((i+2)))
          wrght((i+2))=sqrt(abs(wrght((i+2))))
          if (abs(pstar2((i+2))-hy_plft((i+2)))<small_dp*(pstar2((i+2))+hy_plft((i+2)))) then 

            wlft((i+2))=hy_clft((i+2))
          end if
          wlft((i+2))=max(wlft((i+2)),aux((i+2))*hy_clft((i+2)))
          if (abs(pstar2((i+2))-hy_prght((i+2)))<small_dp*(pstar2((i+2))+hy_prght((i+2)))) then 

            wrght((i+2))=hy_crght((i+2))
          end if
          wrght((i+2))=max(wrght((i+2)),aux((i+2))*hy_crght((i+2)))
          ustrl1=hy_ulft((i+2))-(pstar1((i+2))-hy_plft((i+2)))/wlft1((i+2))
          ustrr1=hy_urght((i+2))+(pstar1((i+2))-hy_prght((i+2)))/wrght1((i+2))
          ustrl2=hy_ulft((i+2))-(pstar2((i+2))-hy_plft((i+2)))/wlft((i+2))
          ustrr2=hy_urght((i+2))+(pstar2((i+2))-hy_prght((i+2)))/wrght((i+2))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+2))=delu2-delu1
          if (abs(pstar2((i+2))-pstar1((i+2)))<=hy_smallp) then 

            scrch1((i+2))=0.0
          end if
          if (abs(scrch1((i+2)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+2))=1.0
          
          end if
          pstar((i+2))=pstar2((i+2))-delu2*(pstar2((i+2))-pstar1((i+2)))/scrch1((i+2))
          pstar((i+2))=max(hy_smallp,pstar((i+2)))
          pres_err=abs(pstar((i+2))-pstar2((i+2)))/pstar((i+2))
          if (pres_err<hy_riemanTol) then 

            goto10((i+2))
          end if
          wlft1((i+2))=wlft((i+2))
          wrght1((i+2))=wrght((i+2))
          pstar1((i+2))=pstar2((i+2))
          pstar2((i+2))=pstar((i+2))
          hy_pstor(n+5)=pstar((i+2))
          gmstrl((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_plft((i+2)))
          gmstrl((i+2))=hy_gmelft((i+2))+2*gmstrl((i+2))/(pstar2((i+2))+hy_plft((i+2)))
          gmstrr((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_prght((i+2)))
          gmstrr((i+2))=hy_gmergt((i+2))+2*gmstrr((i+2))/(pstar2((i+2))+hy_prght((i+2)))
          gmstrl((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrl((i+2))))
          gmstrr((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrr((i+2))))
          scrch1((i+2))=pstar2((i+2))-(gmstrl((i+2))-1)*hy_plft((i+2))/(hy_gmelft((i+2))-1)
          if (scrch1((i+2))==0.0) then 

            scrch1((i+2))=hy_smallp
          end if
          wlft((i+2))=pstar2((i+2))+0.5*(gmstrl((i+2))-1)*(pstar2((i+2))+hy_plft((i+2)))
          wlft((i+2))=(pstar2((i+2))-hy_plft((i+2)))*wlft((i+2))/(hy_vlft((i+2))*scrch1((i+2)))
          wlft((i+2))=sqrt(abs(wlft((i+2))))
          scrch2((i+2))=pstar2((i+2))-(gmstrr((i+2))-1)*hy_prght((i+2))/(hy_gmergt((i+2))-1)
          if (scrch2((i+2))==0.0) then 

            scrch2((i+2))=hy_smallp
          end if
          wrght((i+2))=pstar2((i+2))+0.5*(gmstrr((i+2))-1)*(pstar2((i+2))+hy_prght((i+2)))
          wrght((i+2))=(pstar2((i+2))-hy_prght((i+2)))*wrght((i+2))/(hy_vrght((i+2))*scrch2((i+2)))
          wrght((i+2))=sqrt(abs(wrght((i+2))))
          if (abs(pstar2((i+2))-hy_plft((i+2)))<small_dp*(pstar2((i+2))+hy_plft((i+2)))) then 

            wlft((i+2))=hy_clft((i+2))
          end if
          wlft((i+2))=max(wlft((i+2)),aux((i+2))*hy_clft((i+2)))
          if (abs(pstar2((i+2))-hy_prght((i+2)))<small_dp*(pstar2((i+2))+hy_prght((i+2)))) then 

            wrght((i+2))=hy_crght((i+2))
          end if
          wrght((i+2))=max(wrght((i+2)),aux((i+2))*hy_crght((i+2)))
          ustrl1=hy_ulft((i+2))-(pstar1((i+2))-hy_plft((i+2)))/wlft1((i+2))
          ustrr1=hy_urght((i+2))+(pstar1((i+2))-hy_prght((i+2)))/wrght1((i+2))
          ustrl2=hy_ulft((i+2))-(pstar2((i+2))-hy_plft((i+2)))/wlft((i+2))
          ustrr2=hy_urght((i+2))+(pstar2((i+2))-hy_prght((i+2)))/wrght((i+2))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+2))=delu2-delu1
          if (abs(pstar2((i+2))-pstar1((i+2)))<=hy_smallp) then 

            scrch1((i+2))=0.0
          end if
          if (abs(scrch1((i+2)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+2))=1.0
          
          end if
          pstar((i+2))=pstar2((i+2))-delu2*(pstar2((i+2))-pstar1((i+2)))/scrch1((i+2))
          pstar((i+2))=max(hy_smallp,pstar((i+2)))
          pres_err=abs(pstar((i+2))-pstar2((i+2)))/pstar((i+2))
          if (pres_err<hy_riemanTol) then 

            goto10((i+2))
          end if
          wlft1((i+2))=wlft((i+2))
          wrght1((i+2))=wrght((i+2))
          pstar1((i+2))=pstar2((i+2))
          pstar2((i+2))=pstar((i+2))
          hy_pstor(n+6)=pstar((i+2))
          gmstrl((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_plft((i+2)))
          gmstrl((i+2))=hy_gmelft((i+2))+2*gmstrl((i+2))/(pstar2((i+2))+hy_plft((i+2)))
          gmstrr((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_prght((i+2)))
          gmstrr((i+2))=hy_gmergt((i+2))+2*gmstrr((i+2))/(pstar2((i+2))+hy_prght((i+2)))
          gmstrl((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrl((i+2))))
          gmstrr((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrr((i+2))))
          scrch1((i+2))=pstar2((i+2))-(gmstrl((i+2))-1)*hy_plft((i+2))/(hy_gmelft((i+2))-1)
          if (scrch1((i+2))==0.0) then 

            scrch1((i+2))=hy_smallp
          end if
          wlft((i+2))=pstar2((i+2))+0.5*(gmstrl((i+2))-1)*(pstar2((i+2))+hy_plft((i+2)))
          wlft((i+2))=(pstar2((i+2))-hy_plft((i+2)))*wlft((i+2))/(hy_vlft((i+2))*scrch1((i+2)))
          wlft((i+2))=sqrt(abs(wlft((i+2))))
          scrch2((i+2))=pstar2((i+2))-(gmstrr((i+2))-1)*hy_prght((i+2))/(hy_gmergt((i+2))-1)
          if (scrch2((i+2))==0.0) then 

            scrch2((i+2))=hy_smallp
          end if
          wrght((i+2))=pstar2((i+2))+0.5*(gmstrr((i+2))-1)*(pstar2((i+2))+hy_prght((i+2)))
          wrght((i+2))=(pstar2((i+2))-hy_prght((i+2)))*wrght((i+2))/(hy_vrght((i+2))*scrch2((i+2)))
          wrght((i+2))=sqrt(abs(wrght((i+2))))
          if (abs(pstar2((i+2))-hy_plft((i+2)))<small_dp*(pstar2((i+2))+hy_plft((i+2)))) then 

            wlft((i+2))=hy_clft((i+2))
          end if
          wlft((i+2))=max(wlft((i+2)),aux((i+2))*hy_clft((i+2)))
          if (abs(pstar2((i+2))-hy_prght((i+2)))<small_dp*(pstar2((i+2))+hy_prght((i+2)))) then 

            wrght((i+2))=hy_crght((i+2))
          end if
          wrght((i+2))=max(wrght((i+2)),aux((i+2))*hy_crght((i+2)))
          ustrl1=hy_ulft((i+2))-(pstar1((i+2))-hy_plft((i+2)))/wlft1((i+2))
          ustrr1=hy_urght((i+2))+(pstar1((i+2))-hy_prght((i+2)))/wrght1((i+2))
          ustrl2=hy_ulft((i+2))-(pstar2((i+2))-hy_plft((i+2)))/wlft((i+2))
          ustrr2=hy_urght((i+2))+(pstar2((i+2))-hy_prght((i+2)))/wrght((i+2))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+2))=delu2-delu1
          if (abs(pstar2((i+2))-pstar1((i+2)))<=hy_smallp) then 

            scrch1((i+2))=0.0
          end if
          if (abs(scrch1((i+2)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+2))=1.0
          
          end if
          pstar((i+2))=pstar2((i+2))-delu2*(pstar2((i+2))-pstar1((i+2)))/scrch1((i+2))
          pstar((i+2))=max(hy_smallp,pstar((i+2)))
          pres_err=abs(pstar((i+2))-pstar2((i+2)))/pstar((i+2))
          if (pres_err<hy_riemanTol) then 

            goto10((i+2))
          end if
          wlft1((i+2))=wlft((i+2))
          wrght1((i+2))=wrght((i+2))
          pstar1((i+2))=pstar2((i+2))
          pstar2((i+2))=pstar((i+2))
          hy_pstor(n+7)=pstar((i+2))
          gmstrl((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_plft((i+2)))
          gmstrl((i+2))=hy_gmelft((i+2))+2*gmstrl((i+2))/(pstar2((i+2))+hy_plft((i+2)))
          gmstrr((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_prght((i+2)))
          gmstrr((i+2))=hy_gmergt((i+2))+2*gmstrr((i+2))/(pstar2((i+2))+hy_prght((i+2)))
          gmstrl((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrl((i+2))))
          gmstrr((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrr((i+2))))
          scrch1((i+2))=pstar2((i+2))-(gmstrl((i+2))-1)*hy_plft((i+2))/(hy_gmelft((i+2))-1)
          if (scrch1((i+2))==0.0) then 

            scrch1((i+2))=hy_smallp
          end if
          wlft((i+2))=pstar2((i+2))+0.5*(gmstrl((i+2))-1)*(pstar2((i+2))+hy_plft((i+2)))
          wlft((i+2))=(pstar2((i+2))-hy_plft((i+2)))*wlft((i+2))/(hy_vlft((i+2))*scrch1((i+2)))
          wlft((i+2))=sqrt(abs(wlft((i+2))))
          scrch2((i+2))=pstar2((i+2))-(gmstrr((i+2))-1)*hy_prght((i+2))/(hy_gmergt((i+2))-1)
          if (scrch2((i+2))==0.0) then 

            scrch2((i+2))=hy_smallp
          end if
          wrght((i+2))=pstar2((i+2))+0.5*(gmstrr((i+2))-1)*(pstar2((i+2))+hy_prght((i+2)))
          wrght((i+2))=(pstar2((i+2))-hy_prght((i+2)))*wrght((i+2))/(hy_vrght((i+2))*scrch2((i+2)))
          wrght((i+2))=sqrt(abs(wrght((i+2))))
          if (abs(pstar2((i+2))-hy_plft((i+2)))<small_dp*(pstar2((i+2))+hy_plft((i+2)))) then 

            wlft((i+2))=hy_clft((i+2))
          end if
          wlft((i+2))=max(wlft((i+2)),aux((i+2))*hy_clft((i+2)))
          if (abs(pstar2((i+2))-hy_prght((i+2)))<small_dp*(pstar2((i+2))+hy_prght((i+2)))) then 

            wrght((i+2))=hy_crght((i+2))
          end if
          wrght((i+2))=max(wrght((i+2)),aux((i+2))*hy_crght((i+2)))
          ustrl1=hy_ulft((i+2))-(pstar1((i+2))-hy_plft((i+2)))/wlft1((i+2))
          ustrr1=hy_urght((i+2))+(pstar1((i+2))-hy_prght((i+2)))/wrght1((i+2))
          ustrl2=hy_ulft((i+2))-(pstar2((i+2))-hy_plft((i+2)))/wlft((i+2))
          ustrr2=hy_urght((i+2))+(pstar2((i+2))-hy_prght((i+2)))/wrght((i+2))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+2))=delu2-delu1
          if (abs(pstar2((i+2))-pstar1((i+2)))<=hy_smallp) then 

            scrch1((i+2))=0.0
          end if
          if (abs(scrch1((i+2)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+2))=1.0
          
          end if
          pstar((i+2))=pstar2((i+2))-delu2*(pstar2((i+2))-pstar1((i+2)))/scrch1((i+2))
          pstar((i+2))=max(hy_smallp,pstar((i+2)))
          pres_err=abs(pstar((i+2))-pstar2((i+2)))/pstar((i+2))
          if (pres_err<hy_riemanTol) then 

            goto10((i+2))
          end if
          wlft1((i+2))=wlft((i+2))
          wrght1((i+2))=wrght((i+2))
          pstar1((i+2))=pstar2((i+2))
          pstar2((i+2))=pstar((i+2))
          hy_pstor(n+8)=pstar((i+2))
          gmstrl((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_plft((i+2)))
          gmstrl((i+2))=hy_gmelft((i+2))+2*gmstrl((i+2))/(pstar2((i+2))+hy_plft((i+2)))
          gmstrr((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_prght((i+2)))
          gmstrr((i+2))=hy_gmergt((i+2))+2*gmstrr((i+2))/(pstar2((i+2))+hy_prght((i+2)))
          gmstrl((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrl((i+2))))
          gmstrr((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrr((i+2))))
          scrch1((i+2))=pstar2((i+2))-(gmstrl((i+2))-1)*hy_plft((i+2))/(hy_gmelft((i+2))-1)
          if (scrch1((i+2))==0.0) then 

            scrch1((i+2))=hy_smallp
          end if
          wlft((i+2))=pstar2((i+2))+0.5*(gmstrl((i+2))-1)*(pstar2((i+2))+hy_plft((i+2)))
          wlft((i+2))=(pstar2((i+2))-hy_plft((i+2)))*wlft((i+2))/(hy_vlft((i+2))*scrch1((i+2)))
          wlft((i+2))=sqrt(abs(wlft((i+2))))
          scrch2((i+2))=pstar2((i+2))-(gmstrr((i+2))-1)*hy_prght((i+2))/(hy_gmergt((i+2))-1)
          if (scrch2((i+2))==0.0) then 

            scrch2((i+2))=hy_smallp
          end if
          wrght((i+2))=pstar2((i+2))+0.5*(gmstrr((i+2))-1)*(pstar2((i+2))+hy_prght((i+2)))
          wrght((i+2))=(pstar2((i+2))-hy_prght((i+2)))*wrght((i+2))/(hy_vrght((i+2))*scrch2((i+2)))
          wrght((i+2))=sqrt(abs(wrght((i+2))))
          if (abs(pstar2((i+2))-hy_plft((i+2)))<small_dp*(pstar2((i+2))+hy_plft((i+2)))) then 

            wlft((i+2))=hy_clft((i+2))
          end if
          wlft((i+2))=max(wlft((i+2)),aux((i+2))*hy_clft((i+2)))
          if (abs(pstar2((i+2))-hy_prght((i+2)))<small_dp*(pstar2((i+2))+hy_prght((i+2)))) then 

            wrght((i+2))=hy_crght((i+2))
          end if
          wrght((i+2))=max(wrght((i+2)),aux((i+2))*hy_crght((i+2)))
          ustrl1=hy_ulft((i+2))-(pstar1((i+2))-hy_plft((i+2)))/wlft1((i+2))
          ustrr1=hy_urght((i+2))+(pstar1((i+2))-hy_prght((i+2)))/wrght1((i+2))
          ustrl2=hy_ulft((i+2))-(pstar2((i+2))-hy_plft((i+2)))/wlft((i+2))
          ustrr2=hy_urght((i+2))+(pstar2((i+2))-hy_prght((i+2)))/wrght((i+2))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+2))=delu2-delu1
          if (abs(pstar2((i+2))-pstar1((i+2)))<=hy_smallp) then 

            scrch1((i+2))=0.0
          end if
          if (abs(scrch1((i+2)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+2))=1.0
          
          end if
          pstar((i+2))=pstar2((i+2))-delu2*(pstar2((i+2))-pstar1((i+2)))/scrch1((i+2))
          pstar((i+2))=max(hy_smallp,pstar((i+2)))
          pres_err=abs(pstar((i+2))-pstar2((i+2)))/pstar((i+2))
          if (pres_err<hy_riemanTol) then 

            goto10((i+2))
          end if
          wlft1((i+2))=wlft((i+2))
          wrght1((i+2))=wrght((i+2))
          pstar1((i+2))=pstar2((i+2))
          pstar2((i+2))=pstar((i+2))
          hy_pstor(n+9)=pstar((i+2))
          gmstrl((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_plft((i+2)))
          gmstrl((i+2))=hy_gmelft((i+2))+2*gmstrl((i+2))/(pstar2((i+2))+hy_plft((i+2)))
          gmstrr((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_prght((i+2)))
          gmstrr((i+2))=hy_gmergt((i+2))+2*gmstrr((i+2))/(pstar2((i+2))+hy_prght((i+2)))
          gmstrl((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrl((i+2))))
          gmstrr((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrr((i+2))))
          scrch1((i+2))=pstar2((i+2))-(gmstrl((i+2))-1)*hy_plft((i+2))/(hy_gmelft((i+2))-1)
          if (scrch1((i+2))==0.0) then 

            scrch1((i+2))=hy_smallp
          end if
          wlft((i+2))=pstar2((i+2))+0.5*(gmstrl((i+2))-1)*(pstar2((i+2))+hy_plft((i+2)))
          wlft((i+2))=(pstar2((i+2))-hy_plft((i+2)))*wlft((i+2))/(hy_vlft((i+2))*scrch1((i+2)))
          wlft((i+2))=sqrt(abs(wlft((i+2))))
          scrch2((i+2))=pstar2((i+2))-(gmstrr((i+2))-1)*hy_prght((i+2))/(hy_gmergt((i+2))-1)
          if (scrch2((i+2))==0.0) then 

            scrch2((i+2))=hy_smallp
          end if
          wrght((i+2))=pstar2((i+2))+0.5*(gmstrr((i+2))-1)*(pstar2((i+2))+hy_prght((i+2)))
          wrght((i+2))=(pstar2((i+2))-hy_prght((i+2)))*wrght((i+2))/(hy_vrght((i+2))*scrch2((i+2)))
          wrght((i+2))=sqrt(abs(wrght((i+2))))
          if (abs(pstar2((i+2))-hy_plft((i+2)))<small_dp*(pstar2((i+2))+hy_plft((i+2)))) then 

            wlft((i+2))=hy_clft((i+2))
          end if
          wlft((i+2))=max(wlft((i+2)),aux((i+2))*hy_clft((i+2)))
          if (abs(pstar2((i+2))-hy_prght((i+2)))<small_dp*(pstar2((i+2))+hy_prght((i+2)))) then 

            wrght((i+2))=hy_crght((i+2))
          end if
          wrght((i+2))=max(wrght((i+2)),aux((i+2))*hy_crght((i+2)))
          ustrl1=hy_ulft((i+2))-(pstar1((i+2))-hy_plft((i+2)))/wlft1((i+2))
          ustrr1=hy_urght((i+2))+(pstar1((i+2))-hy_prght((i+2)))/wrght1((i+2))
          ustrl2=hy_ulft((i+2))-(pstar2((i+2))-hy_plft((i+2)))/wlft((i+2))
          ustrr2=hy_urght((i+2))+(pstar2((i+2))-hy_prght((i+2)))/wrght((i+2))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+2))=delu2-delu1
          if (abs(pstar2((i+2))-pstar1((i+2)))<=hy_smallp) then 

            scrch1((i+2))=0.0
          end if
          if (abs(scrch1((i+2)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+2))=1.0
          
          end if
          pstar((i+2))=pstar2((i+2))-delu2*(pstar2((i+2))-pstar1((i+2)))/scrch1((i+2))
          pstar((i+2))=max(hy_smallp,pstar((i+2)))
          pres_err=abs(pstar((i+2))-pstar2((i+2)))/pstar((i+2))
          if (pres_err<hy_riemanTol) then 

            goto10((i+2))
          end if
          wlft1((i+2))=wlft((i+2))
          wrght1((i+2))=wrght((i+2))
          pstar1((i+2))=pstar2((i+2))
          pstar2((i+2))=pstar((i+2))
          hy_pstor(n+10)=pstar((i+2))
          gmstrl((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_plft((i+2)))
          gmstrl((i+2))=hy_gmelft((i+2))+2*gmstrl((i+2))/(pstar2((i+2))+hy_plft((i+2)))
          gmstrr((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_prght((i+2)))
          gmstrr((i+2))=hy_gmergt((i+2))+2*gmstrr((i+2))/(pstar2((i+2))+hy_prght((i+2)))
          gmstrl((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrl((i+2))))
          gmstrr((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrr((i+2))))
          scrch1((i+2))=pstar2((i+2))-(gmstrl((i+2))-1)*hy_plft((i+2))/(hy_gmelft((i+2))-1)
          if (scrch1((i+2))==0.0) then 

            scrch1((i+2))=hy_smallp
          end if
          wlft((i+2))=pstar2((i+2))+0.5*(gmstrl((i+2))-1)*(pstar2((i+2))+hy_plft((i+2)))
          wlft((i+2))=(pstar2((i+2))-hy_plft((i+2)))*wlft((i+2))/(hy_vlft((i+2))*scrch1((i+2)))
          wlft((i+2))=sqrt(abs(wlft((i+2))))
          scrch2((i+2))=pstar2((i+2))-(gmstrr((i+2))-1)*hy_prght((i+2))/(hy_gmergt((i+2))-1)
          if (scrch2((i+2))==0.0) then 

            scrch2((i+2))=hy_smallp
          end if
          wrght((i+2))=pstar2((i+2))+0.5*(gmstrr((i+2))-1)*(pstar2((i+2))+hy_prght((i+2)))
          wrght((i+2))=(pstar2((i+2))-hy_prght((i+2)))*wrght((i+2))/(hy_vrght((i+2))*scrch2((i+2)))
          wrght((i+2))=sqrt(abs(wrght((i+2))))
          if (abs(pstar2((i+2))-hy_plft((i+2)))<small_dp*(pstar2((i+2))+hy_plft((i+2)))) then 

            wlft((i+2))=hy_clft((i+2))
          end if
          wlft((i+2))=max(wlft((i+2)),aux((i+2))*hy_clft((i+2)))
          if (abs(pstar2((i+2))-hy_prght((i+2)))<small_dp*(pstar2((i+2))+hy_prght((i+2)))) then 

            wrght((i+2))=hy_crght((i+2))
          end if
          wrght((i+2))=max(wrght((i+2)),aux((i+2))*hy_crght((i+2)))
          ustrl1=hy_ulft((i+2))-(pstar1((i+2))-hy_plft((i+2)))/wlft1((i+2))
          ustrr1=hy_urght((i+2))+(pstar1((i+2))-hy_prght((i+2)))/wrght1((i+2))
          ustrl2=hy_ulft((i+2))-(pstar2((i+2))-hy_plft((i+2)))/wlft((i+2))
          ustrr2=hy_urght((i+2))+(pstar2((i+2))-hy_prght((i+2)))/wrght((i+2))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+2))=delu2-delu1
          if (abs(pstar2((i+2))-pstar1((i+2)))<=hy_smallp) then 

            scrch1((i+2))=0.0
          end if
          if (abs(scrch1((i+2)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+2))=1.0
          
          end if
          pstar((i+2))=pstar2((i+2))-delu2*(pstar2((i+2))-pstar1((i+2)))/scrch1((i+2))
          pstar((i+2))=max(hy_smallp,pstar((i+2)))
          pres_err=abs(pstar((i+2))-pstar2((i+2)))/pstar((i+2))
          if (pres_err<hy_riemanTol) then 

            goto10((i+2))
          end if
          wlft1((i+2))=wlft((i+2))
          wrght1((i+2))=wrght((i+2))
          pstar1((i+2))=pstar2((i+2))
          pstar2((i+2))=pstar((i+2))
          hy_pstor(n+11)=pstar((i+2))
        

        end do
        do n=(min(hy_nriem,nn+255))-(MOD((min(hy_nriem,nn+255)), 10)), min(hy_nriem,nn+255), +1        
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+2)=pstar(i)
          gmstrl((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_plft((i+1)))
          gmstrl((i+1))=hy_gmelft((i+1))+2*gmstrl((i+1))/(pstar2((i+1))+hy_plft((i+1)))
          gmstrr((i+1))=gamfac((i+1))*(pstar2((i+1))-hy_prght((i+1)))
          gmstrr((i+1))=hy_gmergt((i+1))+2*gmstrr((i+1))/(pstar2((i+1))+hy_prght((i+1)))
          gmstrl((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrl((i+1))))
          gmstrr((i+1))=max(gmin((i+1)),min(gmax((i+1)),gmstrr((i+1))))
          scrch1((i+1))=pstar2((i+1))-(gmstrl((i+1))-1)*hy_plft((i+1))/(hy_gmelft((i+1))-1)
          if (scrch1((i+1))==0.0) then 

            scrch1((i+1))=hy_smallp
          end if
          wlft((i+1))=pstar2((i+1))+0.5*(gmstrl((i+1))-1)*(pstar2((i+1))+hy_plft((i+1)))
          wlft((i+1))=(pstar2((i+1))-hy_plft((i+1)))*wlft((i+1))/(hy_vlft((i+1))*scrch1((i+1)))
          wlft((i+1))=sqrt(abs(wlft((i+1))))
          scrch2((i+1))=pstar2((i+1))-(gmstrr((i+1))-1)*hy_prght((i+1))/(hy_gmergt((i+1))-1)
          if (scrch2((i+1))==0.0) then 

            scrch2((i+1))=hy_smallp
          end if
          wrght((i+1))=pstar2((i+1))+0.5*(gmstrr((i+1))-1)*(pstar2((i+1))+hy_prght((i+1)))
          wrght((i+1))=(pstar2((i+1))-hy_prght((i+1)))*wrght((i+1))/(hy_vrght((i+1))*scrch2((i+1)))
          wrght((i+1))=sqrt(abs(wrght((i+1))))
          if (abs(pstar2((i+1))-hy_plft((i+1)))<small_dp*(pstar2((i+1))+hy_plft((i+1)))) then 

            wlft((i+1))=hy_clft((i+1))
          end if
          wlft((i+1))=max(wlft((i+1)),aux((i+1))*hy_clft((i+1)))
          if (abs(pstar2((i+1))-hy_prght((i+1)))<small_dp*(pstar2((i+1))+hy_prght((i+1)))) then 

            wrght((i+1))=hy_crght((i+1))
          end if
          wrght((i+1))=max(wrght((i+1)),aux((i+1))*hy_crght((i+1)))
          ustrl1=hy_ulft((i+1))-(pstar1((i+1))-hy_plft((i+1)))/wlft1((i+1))
          ustrr1=hy_urght((i+1))+(pstar1((i+1))-hy_prght((i+1)))/wrght1((i+1))
          ustrl2=hy_ulft((i+1))-(pstar2((i+1))-hy_plft((i+1)))/wlft((i+1))
          ustrr2=hy_urght((i+1))+(pstar2((i+1))-hy_prght((i+1)))/wrght((i+1))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+1))=delu2-delu1
          if (abs(pstar2((i+1))-pstar1((i+1)))<=hy_smallp) then 

            scrch1((i+1))=0.0
          end if
          if (abs(scrch1((i+1)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+1))=1.0
          
          end if
          pstar((i+1))=pstar2((i+1))-delu2*(pstar2((i+1))-pstar1((i+1)))/scrch1((i+1))
          pstar((i+1))=max(hy_smallp,pstar((i+1)))
          pres_err=abs(pstar((i+1))-pstar2((i+1)))/pstar((i+1))
          if (pres_err<hy_riemanTol) then 

            goto10((i+1))
          end if
          wlft1((i+1))=wlft((i+1))
          wrght1((i+1))=wrght((i+1))
          pstar1((i+1))=pstar2((i+1))
          pstar2((i+1))=pstar((i+1))
          hy_pstor(n+2)=pstar((i+1))
          gmstrl((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_plft((i+2)))
          gmstrl((i+2))=hy_gmelft((i+2))+2*gmstrl((i+2))/(pstar2((i+2))+hy_plft((i+2)))
          gmstrr((i+2))=gamfac((i+2))*(pstar2((i+2))-hy_prght((i+2)))
          gmstrr((i+2))=hy_gmergt((i+2))+2*gmstrr((i+2))/(pstar2((i+2))+hy_prght((i+2)))
          gmstrl((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrl((i+2))))
          gmstrr((i+2))=max(gmin((i+2)),min(gmax((i+2)),gmstrr((i+2))))
          scrch1((i+2))=pstar2((i+2))-(gmstrl((i+2))-1)*hy_plft((i+2))/(hy_gmelft((i+2))-1)
          if (scrch1((i+2))==0.0) then 

            scrch1((i+2))=hy_smallp
          end if
          wlft((i+2))=pstar2((i+2))+0.5*(gmstrl((i+2))-1)*(pstar2((i+2))+hy_plft((i+2)))
          wlft((i+2))=(pstar2((i+2))-hy_plft((i+2)))*wlft((i+2))/(hy_vlft((i+2))*scrch1((i+2)))
          wlft((i+2))=sqrt(abs(wlft((i+2))))
          scrch2((i+2))=pstar2((i+2))-(gmstrr((i+2))-1)*hy_prght((i+2))/(hy_gmergt((i+2))-1)
          if (scrch2((i+2))==0.0) then 

            scrch2((i+2))=hy_smallp
          end if
          wrght((i+2))=pstar2((i+2))+0.5*(gmstrr((i+2))-1)*(pstar2((i+2))+hy_prght((i+2)))
          wrght((i+2))=(pstar2((i+2))-hy_prght((i+2)))*wrght((i+2))/(hy_vrght((i+2))*scrch2((i+2)))
          wrght((i+2))=sqrt(abs(wrght((i+2))))
          if (abs(pstar2((i+2))-hy_plft((i+2)))<small_dp*(pstar2((i+2))+hy_plft((i+2)))) then 

            wlft((i+2))=hy_clft((i+2))
          end if
          wlft((i+2))=max(wlft((i+2)),aux((i+2))*hy_clft((i+2)))
          if (abs(pstar2((i+2))-hy_prght((i+2)))<small_dp*(pstar2((i+2))+hy_prght((i+2)))) then 

            wrght((i+2))=hy_crght((i+2))
          end if
          wrght((i+2))=max(wrght((i+2)),aux((i+2))*hy_crght((i+2)))
          ustrl1=hy_ulft((i+2))-(pstar1((i+2))-hy_plft((i+2)))/wlft1((i+2))
          ustrr1=hy_urght((i+2))+(pstar1((i+2))-hy_prght((i+2)))/wrght1((i+2))
          ustrl2=hy_ulft((i+2))-(pstar2((i+2))-hy_plft((i+2)))/wlft((i+2))
          ustrr2=hy_urght((i+2))+(pstar2((i+2))-hy_prght((i+2)))/wrght((i+2))
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1((i+2))=delu2-delu1
          if (abs(pstar2((i+2))-pstar1((i+2)))<=hy_smallp) then 

            scrch1((i+2))=0.0
          end if
          if (abs(scrch1((i+2)))<hy_smallu) then 
          
            delu2=0.0
            scrch1((i+2))=1.0
          
          end if
          pstar((i+2))=pstar2((i+2))-delu2*(pstar2((i+2))-pstar1((i+2)))/scrch1((i+2))
          pstar((i+2))=max(hy_smallp,pstar((i+2)))
          pres_err=abs(pstar((i+2))-pstar2((i+2)))/pstar((i+2))
          if (pres_err<hy_riemanTol) then 

            goto10((i+2))
          end if
          wlft1((i+2))=wlft((i+2))
          wrght1((i+2))=wrght((i+2))
          pstar1((i+2))=pstar2((i+2))
          pstar2((i+2))=pstar((i+2))
          hy_pstor(n+2)=pstar((i+2))
        

        end do
      

      end do
      n=n-1
      n=n-1
      n=n-1
      abortcode()
      abortcode()
      abortcode()
    

    end do
    do i=(min(numIntCells5,ii+15))-(MOD((min(numIntCells5,ii+15)), 3)), min(numIntCells5,ii+15), +1    
      hy_pstor(1)=pstar1(i)
      hy_pstor(2)=pstar2(i)
      do nn=1, hy_nriem, +256      
        do n=nn, min(hy_nriem,nn+255)-9, +10        
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+2)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+3)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+4)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+5)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+6)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+7)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+8)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+9)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+10)=pstar(i)
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1)*hy_plft(i)/(hy_gmelft(i)-1)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1)*hy_prght(i)/(hy_gmergt(i)-1)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+11)=pstar(i)
        

        end do
        do n=(min(hy_nriem,nn+255))-(MOD((min(hy_nriem,nn+255)), 10)), min(hy_nriem,nn+255), +1        
          gmstrl(i)=gamfac(i)*(pstar2(i)-hy_plft(i))
          gmstrl(i)=hy_gmelft(i)+2.0*gmstrl(i)/(pstar2(i)+hy_plft(i))
          gmstrr(i)=gamfac(i)*(pstar2(i)-hy_prght(i))
          gmstrr(i)=hy_gmergt(i)+2.0*gmstrr(i)/(pstar2(i)+hy_prght(i))
          gmstrl(i)=max(gmin(i),min(gmax(i),gmstrl(i)))
          gmstrr(i)=max(gmin(i),min(gmax(i),gmstrr(i)))
          scrch1(i)=pstar2(i)-(gmstrl(i)-1.0)*hy_plft(i)/(hy_gmelft(i)-1.0)
          if (scrch1(i)==0.0) then 

            scrch1(i)=hy_smallp
          end if
          wlft(i)=pstar2(i)+0.5*(gmstrl(i)-1.0)*(pstar2(i)+hy_plft(i))
          wlft(i)=(pstar2(i)-hy_plft(i))*wlft(i)/(hy_vlft(i)*scrch1(i))
          wlft(i)=sqrt(abs(wlft(i)))
          scrch2(i)=pstar2(i)-(gmstrr(i)-1.0)*hy_prght(i)/(hy_gmergt(i)-1.0)
          if (scrch2(i)==0.0) then 

            scrch2(i)=hy_smallp
          end if
          wrght(i)=pstar2(i)+0.5*(gmstrr(i)-1.0)*(pstar2(i)+hy_prght(i))
          wrght(i)=(pstar2(i)-hy_prght(i))*wrght(i)/(hy_vrght(i)*scrch2(i))
          wrght(i)=sqrt(abs(wrght(i)))
          if (abs(pstar2(i)-hy_plft(i))<small_dp*(pstar2(i)+hy_plft(i))) then 

            wlft(i)=hy_clft(i)
          end if
          wlft(i)=max(wlft(i),aux(i)*hy_clft(i))
          if (abs(pstar2(i)-hy_prght(i))<small_dp*(pstar2(i)+hy_prght(i))) then 

            wrght(i)=hy_crght(i)
          end if
          wrght(i)=max(wrght(i),aux(i)*hy_crght(i))
          ustrl1=hy_ulft(i)-(pstar1(i)-hy_plft(i))/wlft1(i)
          ustrr1=hy_urght(i)+(pstar1(i)-hy_prght(i))/wrght1(i)
          ustrl2=hy_ulft(i)-(pstar2(i)-hy_plft(i))/wlft(i)
          ustrr2=hy_urght(i)+(pstar2(i)-hy_prght(i))/wrght(i)
          delu1=ustrl1-ustrr1
          delu2=ustrl2-ustrr2
          scrch1(i)=delu2-delu1
          if (abs(pstar2(i)-pstar1(i))<=hy_smallp) then 

            scrch1(i)=0.0
          end if
          if (abs(scrch1(i))<hy_smallu) then 
          
            delu2=0.0
            scrch1(i)=1.0
          
          end if
          pstar(i)=pstar2(i)-delu2*(pstar2(i)-pstar1(i))/scrch1(i)
          pstar(i)=max(hy_smallp,pstar(i))
          pres_err=abs(pstar(i)-pstar2(i))/pstar(i)
          if (pres_err<hy_riemanTol) then 

            goto10(i)
          end if
          wlft1(i)=wlft(i)
          wrght1(i)=wrght(i)
          pstar1(i)=pstar2(i)
          pstar2(i)=pstar(i)
          hy_pstor(n+2)=pstar(i)
        

        end do
      

      end do
      n=n-1
      abortcode()
    

    end do
  

  end do

end do
do i=5, numIntCells5, +1
  scrch3(i)=hy_ulft(i)-(pstar(i)-hy_plft(i))/wlft(i)
  scrch4(i)=hy_urght(i)+(pstar(i)-hy_prght(i))/wrght(i)
  ustar(i)=0.5*(scrch3(i)+scrch4(i))
  urell(i)=ustar(i)-ugrdl(i)
  scrch1(i)=sign(one,urell(i))
  scrch2(i)=0.5*(1.0+scrch1(i))
  scrch3(i)=0.5*(1.0-scrch1(i))
  ps(i)=hy_plft(i)*scrch2(i)+hy_prght(i)*scrch3(i)
  us(i)=hy_ulft(i)*scrch2(i)+hy_urght(i)*scrch3(i)
  uts(i)=hy_utlft(i)*scrch2(i)+hy_utrght(i)*scrch3(i)
  utts(i)=hy_uttlft(i)*scrch2(i)+hy_uttrgt(i)*scrch3(i)
  vs(i)=hy_vlft(i)*scrch2(i)+hy_vrght(i)*scrch3(i)
  games(i)=hy_gmelft(i)*scrch2(i)+hy_gmergt(i)*scrch3(i)
  gamcs(i)=hy_gmclft(i)*scrch2(i)+hy_gmcrgt(i)*scrch3(i)
  rhos(i)=1.0/vs(i)
  rhos(i)=max(hy_smlrho,rhos(i))
  vs(i)=1.0/rhos(i)
  ws(i)=wlft(i)*scrch2(i)+wrght(i)*scrch3(i)
  ces(i)=sqrt(gamcs(i)*ps(i)*vs(i))
  vstar(i)=vs(i)-(pstar(i)-ps(i))/ws(i)/ws(i)
  rhostr(i)=1.0/vstar(i)
  cestar(i)=sqrt(gamcs(i)*pstar(i)*vstar(i))
  wes(i)=ces(i)-scrch1(i)*us(i)
  westar(i)=cestar(i)-scrch1(i)*ustar(i)
  scrch4(i)=ws(i)*vs(i)-scrch1(i)*us(i)
  if (pstar(i)-ps(i)>0.0) then 
  
    wes(i)=scrch4(i)
    westar(i)=scrch4(i)
  
  end if
  wes(i)=wes(i)+scrch1(i)*ugrdl(i)
  westar(i)=westar(i)+scrch1(i)*ugrdl(i)
  gamfac(i)=(1.0-games(i)/gamcs(i))*(games(i)-1.0)
  gmstar(i)=gamfac(i)*(pstar(i)-ps(i))
  gmstar(i)=games(i)+2.0*gmstar(i)/(pstar(i)+ps(i))
  gmstar(i)=max(gmin(i),min(gmax(i),gmstar(i)))


end do

  do n=1, hy_numXn-3, +4  
    do i=5, numIntCells5, 1    
      xnav(i,n)=hy_xnlft(i,n)*scrch2(i)+hy_xnrght(i,n)*scrch3(i)
      xnav(i,(n+1))=hy_xnlft(i,(n+1))*scrch2(i)+hy_xnrght(i,(n+1))*scrch3(i)
      xnav(i,(n+2))=hy_xnlft(i,(n+2))*scrch2(i)+hy_xnrght(i,(n+2))*scrch3(i)
      xnav(i,(n+3))=hy_xnlft(i,(n+3))*scrch2(i)+hy_xnrght(i,(n+3))*scrch3(i)
    

    end do
  

  end do
  do n=(hy_numXn)-(MOD((hy_numXn), 4)), hy_numXn, +1
    do i=5, numIntCells5, 1    
      xnav(i,n)=hy_xnlft(i,n)*scrch2(i)+hy_xnrght(i,n)*scrch3(i)
    

    end do

  end do

do i=5, numIntCells5, +1
  scrch1(i)=max(wes(i)-westar(i),wes(i)+westar(i),hy_smallu)
  scrch1(i)=(wes(i)+westar(i))/scrch1(i)
  scrch1(i)=0.5*(1.0+scrch1(i))
  scrch2(i)=1.0-scrch1(i)
  rhoav(i)=scrch1(i)*rhostr(i)+scrch2(i)*rhos(i)
  uav(i)=scrch1(i)*ustar(i)+scrch2(i)*us(i)
  utav(i)=uts(i)
  uttav(i)=utts(i)
  pav(i)=scrch1(i)*pstar(i)+scrch2(i)*ps(i)
  gameav(i)=scrch1(i)*gmstar(i)+scrch2(i)*games(i)
  if (westar(i)>0.0) then 
  
    rhoav(i)=rhostr(i)
    uav(i)=ustar(i)
    pav(i)=pstar(i)
    gameav(i)=gmstar(i)
  
  end if
  if (wes(i)<0.0) then 
  
    rhoav(i)=rhos(i)
    uav(i)=us(i)
    pav(i)=ps(i)
    gameav(i)=games(i)
  
  end if
  urell(i)=uav(i)-ugrdl(i)


end do
/*@ end @*/

    
      orio_t_end = getClock()
      orio_delta_time = orio_t_end - orio_t_start
      if (orio_delta_time < orio_min_time) then
          orio_min_time = orio_delta_time
      end if
    
    enddo
    
    write(*,"(A,ES20.13,A)",advance="no") "{'[1, 1, 1, 5, 0, 1, 0, 3, 0, 1, 5, 6, 0, 2, 9, 1]' : ", orio_delta_time, "}"
    
    
    
    contains

    real(double) function getClock()
        implicit none
        integer (kind = 8) clock_count, clock_max, clock_rate
        integer ( kind = 8 ), parameter :: call_num = 100

        call system_clock(clock_count, clock_rate, clock_max)

        getClock = dble(clock_count) / dble(call_num * clock_rate)
    end function

end program main

