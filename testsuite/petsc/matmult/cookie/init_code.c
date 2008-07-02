
void init_input_vars()
{
  int i,j;
  for (i=0; i<=NROWS-1; i++) {
    y[i] = 0.0;
    ii[i] = i*NCOLS;
    for (j=0; j<=NCOLS-1; j++) {
      aa[i*NCOLS+j] = 1.0 * ((i+j+5) % 5 + 1);
      aj[i*NCOLS+j] = j;
    }
  }
  ii[NROWS] = NROWS*NCOLS;
  for (i=0; i<=NCOLS+ADD_ELMS-1; i++) {
    x[i] = 1.0 * ((i+5) % 10 + 1);
  }
}
