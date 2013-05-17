
  double a[NRA][NCA], b[NCA][NCB], c[NRA][NCB];

  /*@ begin Loop(
        transform Composite(

        )

  for (i=0; i<=NRA-1; i++)
    for (j=0; j<=NCA-1; j++)
      a[i][j]= i+j;
  for (i=0; i<=NCA-1; i++)
    for (j=0; j<=NCB-1; j++)
      b[i][j]= i*j;
  for (i=0; i<=NRA-1; i++)
    for (j=0; j<=NCB-1; j++)
      c[i][j]= 0;

  ) @*/

  for (i=0; i<=NRA-1; i++)
    for (j=0; j<=NCA-1; j++)
      a[i][j]= i+j;
  for (i=0; i<=NCA-1; i++)
    for (j=0; j<=NCB-1; j++)
      b[i][j]= i*j;
  for (i=0; i<=NRA-1; i++)
    for (j=0; j<=NCB-1; j++)
      c[i][j]= 0;

  /*@ end @*/
