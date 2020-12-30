/*@ begin PerfTuning (
 def build {
  arg build_command = 'gcc';
 }
 def performance_counter {
  arg method = 'basic timer';
  arg repetitions = 10;
 }
 def performance_params {
  param CFLAGS[] = ['-O0', '-O1', '-O2', '-O3'];
  param UF[] = range(2,17,2);
 }

 def input_params {
  param N[] = [10000,1000000];
 }

 def input_vars {
  decl static double y[N] = 0;
  decl double a1 = random;
  decl double a2 = random;
  decl double a3 = random;
  decl double a4 = random;
  decl static double x1[N] = random;
  decl static double x2[N] = random;
  decl static double x3[N] = random;
  decl static double x4[N] = random;
 }

 def search {
  arg algorithm = 'Mlsearch';
  arg total_runs = 10;
 }
) @*/

int i;

/*@ begin Loop ( 
    transform Unroll(ufactor=UF) 
    for (i=0; i<=N-1; i++)
      y[i] = y[i] + a1*x1[i] + a2*x2[i] + a3*x3[i] + a4*x4[i];
) @*/
for (i=0; i<=N-1; i++)
  y[i] = y[i] + a1*x1[i] + a2*x2[i] + a3*x3[i] + a4*x4[i];
/*@ end @*/

/*@ end @*/

