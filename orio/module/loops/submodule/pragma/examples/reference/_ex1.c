void ex1(int n, double ss, double* a, double* b, double* y) {

  register int i;
#pragma Orio Loops(transform Pragma(pragma_str="omp parallel for"))
{
  #pragma omp parallel for
  for (i=0; i<n; i++ ) {
    y[i]=b[i]+ss*a[i];
  }
}
#pragma Oiro

}
