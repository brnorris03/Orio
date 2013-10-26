void add(int n, double *a, double *b, double *c) {

/*@ begin PerfTuning(
      def performance_params {
        param PR[] = [ "vector always",
                       "vector aligned",
                       "vector unaligned",
                       "vector temporal",
                       "vector nontemporal",
                       "vector nontemporal (a,b,c)",
                       "novector",
                       ""];
        param CFLAGS[] = map(join, product(['', '-no-vec'], ['', '-xhost'], ['', '-restrict']));
      }
      def build {
        arg build_command = 'icc -O3 @CFLAGS';
      }
      def input_params {
        param N[] = [10**8];
      }
      def input_vars {
        decl dynamic double a[N] = random;
        decl dynamic double b[N] = random;
        decl dynamic double c[N] = 0;
      }
) @*/

  register int i;
  int n=N;

/*@ Loops(transform Pragma(pragma_str=PR)) @*/

  for(i=0; i<=n; ++i) {
    c[i] = a[i] + b[i];
  }

/*@ @*/
/*@ @*/
}
