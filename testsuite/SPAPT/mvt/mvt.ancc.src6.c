
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


    param U_I[] = range(1,31);
    param U_J[] = range(1,31);
  }

  def search
  {
    arg algorithm = 'Randomsearch';
    arg total_runs = 10000;
  }
   
  def input_params
  {
    let SIZE = 10000;
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
    unrolljam = (['i','j'],[U_I,U_J])
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


