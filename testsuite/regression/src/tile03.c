
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

  for (ia=0; ia<=NRA-1; ia++)
    for (ja=0; ja<=NCA-1; ja++)
      a[ia][ja] = ia+ja;

  for (ib=0; ib<=NCA-1; ib++)
    for (jb=0; jb<=NCB-1; jb++)
      b[ib][jb]= ib*jb;

  for (ic=0; ic<=NRA-1; ic++)
    for (jc=0; jc<=NCB-1; jc++)
      c[ic][jc]= 0;

  /*@ end @*/
