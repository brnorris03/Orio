

void axpy_5(int N, double *y, 
	    double a1, double *x1, double a2, double *x2, 
	    double a3, double *x3, double a4, double *x4,
	    double a5, double *x5) {

/*@ begin PerfTuning (
 def build {
   arg build_command = 'gfortran -O3 -lm';
 } 
 def performance_counter {
   arg random_seed = 0;
   arg repetitions = 5;
 }
 def performance_params {  
   param UF[] = range(1,11);
#   constraint divisible_by_two = (UF % 2 == 0);
 }
 def input_params {
   param N[] = [1000000];

#   param N[] = range(10000,10000000);
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
   decl int i = 0;
 }
 def search {
   arg algorithm = 'Exhaustive';
#   arg algorithm = 'ChaosGA';
#   arg algorithm = 'Simplex';
#   arg total_runs = 100;
#   arg time_limit = 100;

 }
) @*/

/*@ begin Loop ( 
  transform Unroll(ufactor=UF) 
  for (i=0; i<=N-1; i++)
    y[i]=sqrt(y[i])+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
) @*/

 for (i=0; i<=n-1; i++)
   y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];

/*@ end @*/
/*@ end @*/

}
