/*@ begin PerfTuning (
 def build {
   arg build_command = 'gcc';
 } 
 def performance_counter {
   arg repetitions = 3;
 }
 def performance_params 
 {
    param CFLAGS[] = ['-O0', '-O1', '-O2', '-O3'];
 }
 def input_params {
   param N[] = [1000000];
 }
 def input_vars {
   decl dynamic double x1[N] = random;
   decl dynamic double x2[N] = random;
   decl dynamic double x3[N] = random;
   decl dynamic double x4[N] = random;
   decl dynamic double y[N] = 0;
   decl double a1 = random;
   decl double a2 = random;
   decl double a3 = random;
   decl double a4 = random;
 }
 def search {
   arg algorithm = 'Exhaustive';
 }
 #  import spec Axpy4TuningSpec; 
) @*/

int i;
/*@ begin CHiLL (recipe.txt) @*/
for (i=0; i<=N-1; i++)
  y[i] = y[i] + a1*x1[i] + a2*x2[i] + a3*x3[i] + a4*x4[i];
/*@ end @*/   // CHiLL

/*@ end @*/   // PerfTuning


