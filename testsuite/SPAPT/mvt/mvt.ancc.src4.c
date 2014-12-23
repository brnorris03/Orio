
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

    param T2_I[] = [1,16,32,64,128,256,512];
    param T2_J[] = [1,16,32,64,128,256,512];
    param T2_Ia[] = [1,64,128,256,512,1024,2048];
    param T2_Ja[] = [1,64,128,256,512,1024,2048];

    param U2_I[]  = range(1,31); 
    param U2_J[]  = range(1,31); 

    param RT2_I[] = [1,8,32];
    param RT2_J[] = [1,8,32];


    constraint tileI2 = ((T2_Ia == 1) or ((T2_Ia % T2_I == 0) and (T2_Ia > T2_I )));
    constraint tileJ2 = ((T2_Ja == 1) or ((T2_Ja % T2_J == 0) and (T2_Ja > T2_J )));
    constraint reg_capacity = (RT2_I*RT2_J <= 150);
    constraint unroll_limit = ((U2_I == 1) or (U2_J == 1));
    
   }

  def search
  {
    arg algorithm = 'Randomsearch';
    arg total_runs = 100000;
  }
  
  def input_params
  {
    param M = 2500;
    param N = 2500;
  }

  def input_vars
  {
    decl static double a[M][N] = random;
    decl static double y_1[N] = random;
    decl static double y_2[M] = random;
    decl static double x1[M] = 0;
    decl static double x2[N] = 0;
  }

  def validation {
    arg validation_file = 'validation.c';
  }

            
) @*/

int i, j;
int ii, jj;
int iii, jjj;
int it, jt;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))



/*@ begin Loop(
transform Composite(
      tile = [('i',T2_I,'ii'),('j',T2_J,'jj'),
             (('ii','i'),T2_Ia,'iii'),(('jj','j'),T2_Ja,'jjj')],
      unrolljam = (['i','j'],[U2_I,U2_J]),
      regtile = (['i','j'],[RT2_I,RT2_J])
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


