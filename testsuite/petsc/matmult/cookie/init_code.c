
void init_input_vars()
{
  int i,j;
  for (i=0; i<=NROWS-1; i++) {
    yt[i] = 5;
    for (j=0; j<=NCOLS-1; j++) {
      v1[i*NCOLS+j] = (i+j+5) % 5 + 1;
    }
  }
  for (j=0; j<=NCOLS-1; j++) {
    idx[j] = j + COEF*STEP;
  }
  for (j=0; j<=(NCOLS*STRETCH)-1; j++) {
    x[j] = j % 10 + 1;
  }
}
