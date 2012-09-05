void PackAligned(double* src, double* dest, const int stride, const int nelms, const int count) {

  register int i, j, isrc=0, idest=0;

  /*@ begin Loops(transform Pack()

  for(i=count; i; i--) {
    for(j=nelms; j; j--) {
      dest[idest++] = src[isrc++];
    }
    src += stride;
  }

  ) @*/


  /*@ end @*/
}
