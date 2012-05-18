void axpy5(int n, double *y, double a1, double *x1, double a2, double *x2, double a3, double *x3,
           double a4, double *x4, double a5, double *x5) {
  


/*@ begin PerfTuning(
	def search{
    	  arg algorithm = 'MSimplex';
          arg total_runs = 100;
          arg msimplex_x0 = [2,0];
          arg msimplex_reflection_coef = [1.0];
          arg msimplex_expansion_coef = [2.0];
          arg msimplex_contraction_coef = [0.5];
          arg msimplex_shrinkage_coef = 0.5;
          arg msimplex_search_distance = 1;
          arg msimplex_edge_length = 3;
        }
        def performance_params {
          param UF[] = range(1,9);
          param PA[] = [True, False];
        }
        def input_params {
          param N[] = [100];
        }
        def build {
          arg build_command = 'gcc -O3 -fopenmp -lm';
        }
        def input_vars {
          decl double a1 = random;
          decl double a2 = random;
          decl double a3 = random;
          decl double a4 = random;
          decl double a5 = random;
          decl static double y[N] = random;
          decl static double x1[N] = random;
          decl static double x2[N] = random;
          decl static double x3[N] = random;
          decl static double x4[N] = random;
          decl static double x5[N] = random;
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 3;
        }
  ) @*/  
    register int i;
    register int n=N;
    
/*@ begin Loop (
  transform Unroll(ufactor=UF, parallelize=PA)
    for (i=0; i<=n-1; i++)
      y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
) @*/

    for (i=0; i<=n-1; i++)
	y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];

/*@ end @*/
  /*@ end @*/  
}
