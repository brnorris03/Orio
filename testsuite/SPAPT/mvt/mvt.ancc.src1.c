
/*@ begin PerfTuning (        
  def build
  {
    arg build_command = 'gcc -O3 -fopenmp ';
    arg libs = '-lm -lrt';
  }
   
  def performance_counter         
  {
    arg repetitions = 35;
  }
  
  def performance_params
  {
    param T1_I[] = [1,16,32,64,128,256,512];
    param T1_J[] = [1,16,32,64,128,256,512];
    param T2_I[] = [1,64,128,256,512,1024,2048];
    param T2_J[] = [1,64,128,256,512,1024,2048];

    param ACOPY_A[] = [False,True];

    param U_I[] = range(1,31);
    param U_J[] = range(1,31);

    param SCREP[] = [False,True];
    param VEC[] = [False,True];
    param OMP[] = [False,True];
    
    param PAR1i[]=[False,True];
    param PAR1j[]=[False,True];
    

    constraint tileI = ((T2_I == 1) or (T2_I % T1_I == 0));
    constraint tileJ = ((T2_J == 1) or (T2_J % T1_J == 0));

    constraint reg_capacity = (2*U_I*U_J + 2*U_I + 2*U_J <= 130);

    constraint copy_limitA = ((not ACOPY_A) or (ACOPY_A and 
                              (T1_I if T1_I>1 else T2_I)*(T1_J if T1_J>1 else T2_J) <= 512*512));
  }

  def search
  {
    arg algorithm = 'Randomsearch';
    arg total_runs = 10000;
  }
   
  def input_params
  {
    let SIZE = 2500;
    param MSIZE = SIZE;
    param NSIZE = SIZE;
    param M = SIZE;
    param N = SIZE;
  }

  def input_vars
  {
    decl static double a[M][N] = random;
    decl static double y_1[N] = random;
    decl static double y_2[M] = random;
    decl static double x1[M] = 0;
    decl static double x2[N] = 0;
  }

            
) @*/

int i, j;
int ii, jj;
int iii, jjj;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))



/*@ begin Loop(
  transform Composite(
    tile = [('i',T1_I,'ii'),('j',T1_J,'jj'),(('ii','i'),T2_I,'iii'),(('jj','j'),T2_J,'jjj')],
    arrcopy = [(ACOPY_A,'a[i][j]',[(T1_I if T1_I>1 else T2_I),(T1_J if T1_J>1 else T2_J)],'_copy')],
    scalarreplace = (SCREP, 'double', 'scv_'),
    vector = (VEC, ['ivdep','vector always']),
    unrolljam = (['i','j'],[U_I,U_J]),
    openmp = (OMP, 'omp parallel for private(iii,jjj,ii,jj,i,j,a_copy)')
  )
for (i=0;i<=N-1;i++)
  for (j=0;j<=N-1;j++) 
  { 
    x1[i]=x1[i]+a[i][j]*y_1[j]; 
    x2[j]=x2[j]+a[i][j]*y_2[i]; 
  } 

) @*/


/*@ end @*/
/*@ end @*/


