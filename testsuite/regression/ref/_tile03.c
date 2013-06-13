
  double a[NRA][NCA], b[NCA][NCB], c[NRA][NCB];
  int ia,ja,ib,jb,ic,jc;

  /*@ begin Loop(

  transform Composite(tile = [('ia',32,'tia'), ('ja',32,'tja')])

  for (ia=0; ia<=NRA-1; ia++)
    for (ja=0; ja<=NCA-1; ja++)
      a[ia][ja] = ia+ja;

  transform Composite(tile = [('ib',32,'tib'), ('jb',32,'tjb')])

  for (ib=0; ib<=NCA-1; ib++)
    for (jb=0; jb<=NCB-1; jb++)
      b[ib][jb]= ib*jb;

  transform Composite(tile = [('ic',32,'tic'), ('jc',32,'tjc')])

  for (ic=0; ic<=NRA-1; ic++)
    for (jc=0; jc<=NCB-1; jc++)
      c[ic][jc]= 0;

  ) @*/
  for (tia=0; tia<=NRA-1; tia=tia+32) 
    for (ia=tia; ia<=min(NRA-1,tia+31); ia=ia+1) 
      for (tja=0; tja<=NCA-1; tja=tja+32) 
        for (ja=tja; ja<=min(NCA-1,tja+31); ja=ja+1) 
          a[ia][ja]=ia+ja;
  for (tib=0; tib<=NCA-1; tib=tib+32) 
    for (ib=tib; ib<=min(NCA-1,tib+31); ib=ib+1) 
      for (tjb=0; tjb<=NCB-1; tjb=tjb+32) 
        for (jb=tjb; jb<=min(NCB-1,tjb+31); jb=jb+1) 
          b[ib][jb]=ib*jb;
  for (tic=0; tic<=NRA-1; tic=tic+32) 
    for (ic=tic; ic<=min(NRA-1,tic+31); ic=ic+1) 
      for (tjc=0; tjc<=NCB-1; tjc=tjc+32) 
        for (jc=tjc; jc<=min(NCB-1,tjc+31); jc=jc+1) 
          c[ic][jc]=0;
  /*@ end @*/
