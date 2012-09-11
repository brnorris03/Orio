void PackAligned(double* src, double* dest, const int stride, const int nelms, const int cnt) {

  register int i, j, isrc=0, idest=0;

  /*@ begin Loops(transform Pack(prefetch="src")

  for(i=cnt; i; i--) {
    for(j=nelms; j; j--) {
      dest[idest++] = src[isrc++];
    }
    src += stride;
    isrc = 0;
  }

  ) @*/


  /*@ end @*/
}
