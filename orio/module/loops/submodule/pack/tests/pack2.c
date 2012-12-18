void PackAligned(double* src, double* dest, const int stride, const int nelms, const int cnt) {

  // __builtin_prefetch (&src[isrc]  , 1, 0);
  // __builtin_prefetch (&dest[idest], 0, 0);
  /*@ begin PerfTuning (
        def performance_params {
          param PTRS[]            = [('src')];
          param DIST[]            = range(7,16,8);
          param PF_LATENCY[]      = range(400,401);
          param PF_SIMULT_PFS[]   = range(1,7);
          param PF_L1_LINESZ[]    = range(128,129);
          param PF_L1_CACHESZ[]   = range(32,33);
          param PF_L2_CACHESZ[]   = range(2048,2049);
          param PF_MIN_INSN2PF[]  = range(7,8);
          param PF_MIN_INSN2MEM[] = range(4,5);
          param CFLAGS[] = map(join, product(['','-O1','-O2','-O3']));
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
          arg build_command = 'gcc -fprefetch-loop-arrays'\
                                    ' --param prefetch-latency=@PF_LATENCY@'\
                                    ' --param simultaneous-prefetches=@PF_SIMULT_PFS@'\
                                    ' --param l1-cache-line-size=@PF_L1_LINESZ@'\
                                    ' --param l1-cache-size=@PF_L1_CACHESZ@'\
                                    ' --param l2-cache-size=@PF_L2_CACHESZ@'\
                                    ' --param min-insn-to-prefetch-ratio=@PF_MIN_INSN2PF@'\
                                    ' --param prefetch-min-insn-to-mem-ratio=@PF_MIN_INSN2MEM@'\
                                    ' @CFLAGS';
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
