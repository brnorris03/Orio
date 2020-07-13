/*@ begin PerfTuning (
 def build {
   arg build_command = 'gcc -O3';
   #arg libs = '-lrt';
 } 
 def performance_counter {
   #arg method = 'bgp counter';
   arg repetitions = 3;
 }
 def performance_params {  
   param UF[] = range(1,11);
   param VEC[] = [False,True];
   #constraint divisible_by_two = (UF % 2 == 0);
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
   arg algorithm = 'Randomlocal';
   arg total_runs = 10;
 }
) 
@*/


register int i;

/*@ begin Loop ( 
    transform Composite(
      unrolljam = (['i'],[UF]),
      vector = (VEC, ['ivdep','vector always'])
     )
  for (i=0; i<=N-1; i++)
    y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
) @*/


/*@ end @*/
/*@ end @*/

