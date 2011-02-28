! allocate arrays
    !open (2, FILE='inputs.txt', ACTION='READ')
    !read(2,*)  array_len
    !close(2)
    array_len = N
    allocate(xnav(array_len,HY_NUMXN))
    allocate(hy_xnlft(array_len,HY_NUMXN))
    allocate(hy_xnrght(array_len,HY_NUMXN))


    allocate(aux(array_len)) 
    allocate(game(array_len)) 
    allocate(gmin(array_len))
    allocate(gmax(array_len))
    allocate(gamfac(array_len))
    allocate(hy_gmelft(array_len))
    allocate(hy_gmergt(array_len))
    allocate(hy_gmclft(array_len))
    allocate(hy_gmcrgt(array_len))
    allocate(pstar1(array_len))
    allocate(hy_prght(array_len))
    allocate(hy_plft(array_len))
    allocate(hy_crght(array_len))
    allocate(hy_clft(array_len))
    allocate(hy_ulft(array_len))
    allocate(hy_urght(array_len))
    allocate(gmstrl(array_len))
    allocate(gmstrr(array_len))
    allocate( scrch1(array_len))
    allocate(scrch2(array_len))
    allocate(wlft1(array_len))
    allocate(wrght1(array_len))
    allocate(hy_vrght(array_len))
    allocate(hy_vlft(array_len))
    allocate(pstar2(array_len))
    allocate(wlft(array_len))
    allocate(wrght(array_len))
    allocate(pstar(array_len))
    allocate(scrch3(array_len))
    allocate(scrch4(array_len))
    allocate(ustar(array_len))
    allocate(urell(array_len))
    allocate(ugrdl(array_len))
    allocate(ps(array_len))
    allocate(us(array_len))
    allocate(uts(array_len))
    allocate(utts(array_len))
    allocate(hy_utlft(array_len))
    allocate(hy_utrght(array_len))
    allocate(hy_uttlft(array_len))
    allocate(hy_uttrgt(array_len))
    allocate( vs(array_len))
    allocate(ws(array_len))
    allocate(games(array_len))
    allocate(gamcs(array_len))
    allocate(rhos(array_len))
    allocate(ces(array_len))
    allocate(vstar(array_len))
    allocate(rhostr(array_len))
    allocate(cestar(array_len))
    allocate(wes(array_len))
    allocate(westar(array_len))
    allocate(gmstar(array_len))
    allocate(x(array_len))
    allocate(rhoav(array_len))
    allocate(uav(array_len))
    allocate(utav(array_len))
    allocate(uttav(array_len))
    allocate(pav(array_len))
    allocate(gameav(array_len))


! initialize variables

    call random_seed
    one = 1.0
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
    wlft(:) = 0.0
    wlft1(:) = 0.0
    wrght1(:) = 0.0
    pstar1(:) = 0.0
    pstar2(:) = 0.0
    pstar(:) = 0.0
    !hy_pstor(:) = 0.0
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
    rhoav(:) = 0.0
    uav(:) = 0.0
    utav(:) = 0.0
    uttav(:) = 0.0
    pav(:) = 0.0
    gameav(:) = 0.0
    x(:) = 0.0
    xnav(:,:) = 0.0

    pres_err = 0.0

#if 0
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
#endif

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




    open (2, FILE='inputs.txt', ACTION='READ')
    read(2,*)  array_len
    read(2,*)  numIntCells
    read(2,*)  hy_numXn
    read(2,*) numCells
    read(2,*) numIntCells5
    read(2,*)  hy_numXn
    read(2,*)  hy_smallp 
    read(2,*)  hy_smallu
    read(2,*)  hy_smlrho
    read(2,*)   hy_nriem
    read(2,*)   hy_riemanTol
    ! Arrays
    read(2,*) x
    read(2,*) ugrdl
    read(2,*) game
    read(2,*) hy_gmelft
    read(2,*)  hy_gmergt
    read(2,*)  hy_gmclft
    read(2,*)  hy_gmcrgt
    read(2,*)  hy_prght
    read(2,*)  hy_plft
    read(2,*)  hy_crght
    read(2,*)  hy_clft
    read(2,*)  hy_ulft
    read(2,*)  hy_utlft
    read(2,*)  hy_uttlft
    read(2,*)  hy_urght
    read(2,*)  hy_utrght
    read(2,*)  hy_uttrgt
    read(2,*)  hy_vrght
    read(2,*)  hy_vlft
    read(2,*)  hy_xnlft
    read(2,*)  hy_xnrght
    read(2,*)   hy_gmclft
    read(2,*)  hy_gmcrgt
    allocate(hy_pstor(hy_nriem+2))
    read(2,*)  hy_pstor
    close(2)

    numIntCells = NUMINTCELLS
    numIntCells5 = NUMINTCELLS+5
