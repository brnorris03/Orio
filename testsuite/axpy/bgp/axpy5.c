
void axpy5(int N, double *y, double a1, double *x1, double a2, double *x2, double a3, double *x3, 
	   double a4, double *x4, double a5, double *x5) {

/*@ begin PerfTuning (
 def build {
   arg build_command = 'mpixlc_r -O3 -qstrict -qsmp=omp:noauto';
   arg batch_command = 'qsub -n 32 -t 20 -q short --env "BG_MAXALIGNEXP=0"';
   arg status_command = 'qstat';
   arg num_procs = 32;
 }
 def performance_counter {
   arg method = 'bgp counter';
   arg repetitions = 10;
 }
 def performance_params {  
   param UF[] = range(1,11);
   param PR[] = ['', 'omp parallel for private(i)'];
   param IL[] = [True, False];
   constraint divisible_by_two = (UF % 2 == 0);
   constraint sequential_or_parallel = ((not PR and not IL) or (PR and IL));
 }
 def input_params {
   param N[] = [100000];
 }
 def input_vars {
   decl dynamic double x1[N] = random;
   decl dynamic double x2[N] = random;
   decl dynamic double x3[N] = random;
   decl dynamic double x4[N] = random;
   decl dynamic double x5[N] = random;
   decl dynamic double y[N] = 0;
   decl double a1 = random;
   decl double a2 = random;
   decl double a3 = random;
   decl double a4 = random;
   decl double a5 = random;
 }
 def search {
   arg algorithm = 'Exhaustive';
 }
) @*/

  int n=N;
  register int i;

/*@ begin Align (x1[],x2[],x3[],x4[],x5[],y[]) @*/
/*@ begin Loop (
  transform Pragma(pragma_str=[PR])
  transform Unroll(ufactor=UF, init_cleanup_loop=IL) 
  for (i=0; i<=n-1; i++)
    y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
) @*/

 for (i=0; i<=n-1; i++)
   y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];

/*@ end @*/
/*@ end @*/
/*@ end @*/

}
