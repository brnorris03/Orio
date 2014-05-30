spec unroll_vectorize {
 def build {
   arg build_command = 'gcc -O0';
   arg libs = '-lrt';
 } 
 def performance_counter {
   #arg method = 'bgp counter';
   arg repetitions = 3;
 }
 def performance_params {  
   param UF[] = range(1,11);
   param VEC[] = [False,True];
   param CFLAGS[] = ['-O0', '-O1','-O2','-O3'];
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
}
