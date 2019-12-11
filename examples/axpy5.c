
void axpy_5(int N, double *y, 
	    double a1, double *x1, double a2, double *x2, 
	    double a3, double *x3, double a4, double *x4,
	    double a5, double *x5) {

/*@ begin PerfTuning (
 def build {
   arg build_command = 'cc';
   #arg libs = '-lrt';  # Only needed on linux
 } 
 def performance_counter {
   arg repetitions = 5;
 }
 def performance_params {  
   param UF[] = range(1,11);
   param CFLAGS[] = ['-O0', '-O1', '-O2', '-O3'];
   #constraint divisible_by_two = (UF % 2 == 0);
 }
 def input_params {
   param N[] = [1000000];
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

/*@ begin Loop ( 
  transform Unroll(ufactor=UF) 
  for (i=0; i<=n-1; i++)
    y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
) @*/

 for (i=0; i<=n-1; i++)
   y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];

/*@ end @*/
/*@ end @*/

}
