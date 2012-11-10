void PackAligned(double* src, double* dest, const int stride, const int nelms, const int cnt) {

  // __builtin_prefetch (&src[isrc]  , 1, 0);
  // __builtin_prefetch (&dest[idest], 0, 0);
  /*@ begin PerfTuning (
        def performance_params {
          param PTRS[] = [('src')];
          param DIST[] = range(7,16,8);
          param CFLAGS[] = map(join, product([
                            ' --param prefetch-latency=400',
                            ' --param simultaneous-prefetches=6',
                            ' --param l1-cache-line-size=128'
                            ' --param l1-cache-size=32',
                            ' --param l2-cache-size=2048',
                            ' --param min-insn-to-prefetch-ratio=7',
                            ' --param prefetch-min-insn-to-mem-ratio=4'
              ]));
        }
        def input_params {
          param nelms[]  = range(100,101);
          param cnt[]    = range(150,151);
          param stride[] = range(200,201);
        }
        def input_vars {
          decl dynamic double  src[nelms*cnt*stride] = random;
          decl dynamic double dest[nelms*cnt] = 0;
        }
        def build {
          arg build_command = 'gcc -fprefetch-loop-arrays @CFLAGS';
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 5;
        }
  ) @*/

  register int i, j, isrc=0, idest=0;
  double *init_src=src;

  /*@ begin Loops(transform Pack(prefetch=PTRS, prefetch_distance=DIST)

  for(i=cnt; i; i--) {
    for(j=nelms; j; j--) {
      dest[idest++] = src[isrc++];
    }
    src += stride;
    isrc = 0;
  }

  ) @*/


  /*@ end @*/

  src=init_src;

  /*@ end @*/
}
