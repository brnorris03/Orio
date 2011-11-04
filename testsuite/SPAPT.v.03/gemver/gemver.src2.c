/*@ begin PerfTuning (  
  def build
  {
    arg build_command = 'gcc -O3 -fopenmp -DDYNAMIC';
    arg libs = '-lm -lrt';
  }

  
  def performance_counter  
  { 
  arg repetitions = 35;
  }
  
  def performance_params 
  {

    # Cache tiling
    param T2_I[] = [1,16,32,64,128,256,512];
    param T2_J[] = [1,16,32,64,128,256,512];
    param T2_Ia[] = [1,64,128,256,512,1024,2048];
    param T2_Ja[] = [1,64,128,256,512,1024,2048];
    
    param T4_I[] = [1,16,32,64,128,256,512];
    param T4_J[] = [1,16,32,64,128,256,512];
    param T4_Ia[] = [1,64,128,256,512,1024,2048];
    param T4_Ja[] = [1,64,128,256,512,1024,2048];


    # Unroll-jam 
    param U1_I[]  = range(1,31); 
    param U2_I[]  = [1]; 
    param U2_J[]  = range(1,31); 
    param U3_I[]  = range(1,31);
    param U4_I[]  = range(1,31);
    param U4_J[]  = range(1,31);



    # Register tiling
    param RT2_I[] = [1,8,32];
    param RT2_J[] = [1,8,32];
    param RT4_I[] = [1,8,32];
    param RT4_J[] = [1,8,32];


    # Scalar replacement
    param SCR[]  = [False,True];

    # Vectorization
    param VEC1[] = [False,True];
    param VEC2[] = [False,True];
    param VEC3[] = [False,True];
    param VEC4[] = [False,True];



    # Parallelization
    param OMP[] = [False,True];


    # Constraints
    constraint tileI2 = ((T2_Ia == 1) or (T2_Ia % T2_I == 0));
    constraint tileJ2 = ((T2_Ja == 1) or (T2_Ja % T2_J == 0));
    constraint tileI4 = ((T4_Ia == 1) or (T4_Ia % T4_I == 0));
    constraint tileJ4 = ((T4_Ja == 1) or (T4_Ja % T4_J == 0));

    
    constraint reg_capacity_2 = (RT2_I*RT2_J <= 150);
    constraint reg_capacity_4 = (RT4_I*RT4_J <= 150);
    
    
    constraint unroll_limit_2 = (U2_I == 1) or (U2_J == 1);
    constraint unroll_limit_4 = (U4_I == 1) or (U4_J == 1);

  }
			 
  def search 
  { 
    arg algorithm = 'Randomsearch'; 
    arg total_runs = 10000;
  } 
  
  def input_params 
  {
  param N[] = [20000];
  }
  
  def input_vars
  {
  arg decl_file = 'decl.h';
  arg init_file = 'init.c';
  }
) @*/

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))


int i,j, k,t;
int it, jt, kt;
int ii, jj, kk;
int iii, jjj, kkk;



/*@ begin Loop(

transform Composite(
  unrolljam = (['i'],[U1_I]),
  vector = (VEC1, ['ivdep','vector always'])
)
for (i=0;i<=n-1;i++) {
  x[i]=0;
  w[i]=0;
 }


transform Composite(
    tile = [('j',T2_J,'jj'),('i',T2_I,'ii'),
            (('jj','j'),T2_Ja,'jjj'),(('ii','i'),T2_Ia,'iii')],
    unrolljam = (['j','i'],[U2_J,U2_I]),
    scalarreplace = (SCR, 'double'),
    regtile = (['j','i'],[RT2_J,RT2_I]),
    vector = (VEC2, ['ivdep','vector always']),
    openmp = (OMP, 'omp parallel for private(iii,jjj,kkk,ii,jj,kk,i,j,k)')
)
for (j=0;j<=n-1;j++) {
  for (i=0;i<=n-1;i++) {
    B[j*n+i]=u2[j]*v2[i]+u1[j]*v1[i]+A[j*n+i];
    x[i]=y[j]*B[j*n+i]+x[i];
  }
 }

transform Composite(
  unrolljam = (['i'],[U3_I]),
  vector = (VEC3, ['ivdep','vector always'])
)
for (i=0;i<=n-1;i++) {
  x[i]=b*x[i]+z[i];
 }


transform Composite(
    tile = [('i',T4_I,'ii'),('j',T4_J,'jj'),
            (('ii','i'),T4_Ia,'iii'),(('jj','j'),T4_Ja,'jjj')],
    unrolljam = (['i','j'],[U4_J,U4_I]),
    scalarreplace = (SCR, 'double'),
    regtile = (['i','j'],[RT4_I,RT4_J]),
    vector = (VEC4, ['ivdep','vector always']),
    openmp = (OMP, 'omp parallel for private(iii,jjj,kkk,ii,jj,kk,i,j,k)')
)
for (i = 0; i <= n-1; i++) {
  for (j = 0; j <= n-1; j++) {
    w[i] = w[i] + B[i*n+j]*x[j];
  }
  w[i] = a*w[i];
 }

) @*/

/*@ end @*/

/*@ end @*/



