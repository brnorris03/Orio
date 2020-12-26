void PackAligned(double* src, double* dest, const int stride, const int nelms, const int cnt) {

  /*@ begin PerfTuning (
        def performance_params {
          param PTRS[] = [('src')];
          param CFLAGS[] = ['-fprefetch-loop-arrays'];
        }
        def input_params {
          param nelms[] = range(2,3);
          param cnt[] = range(3,4);
          param stride[] = range(4,5);
        }
        def input_vars {
          decl dynamic double  src[nelms*cnt*stride] = random;
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

  register int i, j;
  double *init_dest=dest, *init_src=src;

  /*@ begin Loops(transform Pack(prefetch=PTRS)

  for(i=cnt; i; i--) {
    for(j=nelms; j; j--) {
      *dest++ = *src++;
    }
    src += stride-nelms;
  }

  ) @*/


  /*@ end @*/

  dest=init_dest;
  src=init_src;

  /*@ end @*/
}
