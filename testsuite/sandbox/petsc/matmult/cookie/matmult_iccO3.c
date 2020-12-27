/*@ begin PerfTuning (  
 def build { 
#   arg build_command = 'gcc -fopenmp -I/disks/soft/papi-3.9.0/include -L/disks/soft/papi-3.9.0/lib64 -lpapi';
   arg build_command = 'icc -openmp -O3 -I/disks/soft/papi-3.9.0/include -L/disks/soft/papi-3.9.0/lib64 -lpapi';
 } 

 def performance_counter {
   arg repetitions = 10;
 }

 def performance_params {
   param U1[] = [1]+list(range(2,10));
   param U2[] = [1]+list(range(2,10));
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

/**-- (Generated by Orio) 
Best performance cost: 
  1755.000000 
Tuned for specific problem sizes: 
  NCOLS = 15 
  NROWS = 15 
Best performance parameters: 
  OMP = False 
  SREP = True 
  U1 = 1 
  U2 = 5 
--**/

 

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
{
  for (i=0; i<=m-1; i++ ) {
    double scv_1;
    scv_1=y[i];
    scv_1=0.0;
    for (jt=ii[i]; jt<=ii[i+1]-5; jt=jt+5) {
      scv_1=scv_1+aa[jt]*x[aj[jt]];
      scv_1=scv_1+aa[(jt+1)]*x[aj[(jt+1)]];
      scv_1=scv_1+aa[(jt+2)]*x[aj[(jt+2)]];
      scv_1=scv_1+aa[(jt+3)]*x[aj[(jt+3)]];
      scv_1=scv_1+aa[(jt+4)]*x[aj[(jt+4)]];
    }
    for (j=jt; j<=ii[i+1]-1; j=j+1) {
      scv_1=scv_1+aa[j]*x[aj[j]];
    }
    y[i]=scv_1;
  }
}
/*@ end @*/

/*@ end @*/

