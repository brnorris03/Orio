/*@ begin PerfTuning (  
 def build { 
   arg build_command = 'gcc -fopenmp -O3 -I/disks/soft/papi-3.9.0/include -L/disks/soft/papi-3.9.0/lib64 -lpapi';
#   arg build_command = 'icc -openmp -O3 -I/disks/soft/papi-3.9.0/include -L/disks/soft/papi-3.9.0/lib64 -lpapi';
 } 

 def performance_counter {
   arg repetitions = 10;
 }

 def performance_params {
   param U1[] = [1]+range(2,10);
   param U2[] = [1]+range(2,10);
   param OMP[] = [False];
   param SREP[] = [False,True];
   constraint par_loop = ((not OMP) or (OMP and U1==1));
 }

 def input_params {
   param NROWS[] = [15];
   param NCOLS[] = [15];
 } 
 
 def input_vars { 
   arg decl_file = 'decl_code.h';
   arg init_file = 'init_code.c'; 
 } 
 
 def performance_test_code { 
   arg skeleton_code_file = 'skeleton_code.c';  
 } 
) @*/ 

int i,j,it,jt;
int m = NROWS;
int newlb_j, newub_j;

/*@ begin Loop (
transform Composite(openmp = (OMP, 'omp parallel for private(i,j,it,jt)'))
transform Composite(scalarreplace = (SREP, 'double'))
transform RegTile(loops=['i','j'], ufactors=[U1,U2])
for (i=0; i<=m-1; i++)
  {
    y[i] = 0.0;
    for (j=ii[i]; j<=ii[i+1]-1; j++)
      y[i] = y[i] + aa[j] * x[aj[j]];
  }
) @*/

for (i=0; i<=m-1; i++)
  {
    y[i] = 0.0;
    for (j=ii[i]; j<=ii[i+1]-1; j++)
      y[i] = y[i] + aa[j] * x[aj[j]];
  }

/*@ end @*/

/*@ end @*/

