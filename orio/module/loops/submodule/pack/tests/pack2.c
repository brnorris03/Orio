void PackAligned(double* src, double* dest, const int stride, const int nelms, const int cnt) {

  /*@ begin PerfTuning (
        def performance_params {
          param PTRS[] = [('src')];
          param CFLAGS[] = map(join, product(['-fprefetch-loop-arrays']));
        }
        def input_params {
          param nelms[] = range(8,9);
          param cnt[] = range(8,9);
          param stride[] = range(1,2);
        }
        def input_vars {
          decl dynamic double  src[nelms*cnt] = random;
          decl dynamic double dest[nelms*cnt] = 0;
        }
        def build {
          arg build_command = 'gcc @CFLAGS';
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 6;
        }
  ) @*/

  register int i, j, isrc=0, idest=0;

  /*@ begin Loops(transform Pack(prefetch=PTRS)

  for(i=cnt; i; i--) {
    for(j=nelms; j; j--) {
      dest[idest++] = src[isrc++];
    }
    src += stride;
    isrc = 0;
  }

  ) @*/


  /*@ end @*/
  /*@ end @*/
}

