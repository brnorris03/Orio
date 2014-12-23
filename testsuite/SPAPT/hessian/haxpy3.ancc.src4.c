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

    # Cache tiling
    param T2_I[] = [1,16,32,64,128,256,512];
    param T2_J[] = [1,16,32,64,128,256,512];
    param T2_Ia[] = [1,64,128,256,512,1024,2048];
    param T2_Ja[] = [1,64,128,256,512,1024,2048];

    param RT_I[] = [1,8,32];
    param RT_J[] = [1,8,32];

    param U_I[] = range(1,31);
    param U_J[] = range(1,31);



    constraint tileI2 = ((T2_Ia == 1) or ((T2_Ia % T2_I == 0) and (T2_Ia > T2_I )));
    constraint tileJ2 = ((T2_Ja == 1) or ((T2_Ja % T2_J == 0) and (T2_Ja > T2_J )));
    constraint reg_capacity = (RT_I*RT_J <= 150);
    constraint unroll_limit = ((U_I == 1) or (U_J == 1));

  }

  def search
  {
    arg algorithm = 'Randomsearch';
    arg total_runs = 100000;
  }
  
  def input_params
  {
    param SIZE = 2000;
    param N = 2000;
  }            

  def input_vars
  { 
    decl static double X0[N][N] = random;
    decl static double X1[N][N] = random;
    decl static double X2[N][N] = random;
    decl static double Y[N][N] = 0;
    decl static double u0[N] = random;
    decl static double u1[N] = random;
    decl static double u2[N] = random;
    decl double a0 = 32.12;
    decl double a1 = 3322.12;
    decl double a2 = 1.123;
    decl double b00 = 1321.9;
    decl double b01 = 21.55;
    decl double b02 = 10.3;
    decl double b11 = 1210.313;
    decl double b12 = 9.373;
    decl double b22 = 1992.31221;
  }            


  def validation {

    arg validation_file = 'validation.c';

  }





) @*/

int i,j,ii,jj,iii,jjj,it,jt;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*@ begin Loop(
transform Composite(
      tile = [('i',T2_I,'ii'),('j',T2_J,'jj'),
             (('ii','i'),T2_Ia,'iii'),(('jj','j'),T2_Ja,'jjj')],
      unrolljam = (['i','j'],[U_I,U_J]),
      regtile = (['i','j'],[RT_I,RT_J])
)
for (i=0; i<=N-1; i++)
  for (j=0; j<=N-1; j++) 
    {
      Y[i][j]=a0*X0[i][j] + a1*X1[i][j] + a2*X2[i][j]
	+ 2.0*b00*u0[i]*u0[j]
	+ 2.0*b11*u1[i]*u1[j]
	+ 2.0*b22*u2[i]*u2[j]
	+ b01*u0[i]*u1[j] + b01*u1[i]*u0[j] 
	+ b02*u0[i]*u2[j] + b02*u2[i]*u0[j]
	+ b12*u1[i]*u2[j] + b12*u2[i]*u1[j];
    }

) @*/

/*@ end @*/
/*@ end @*/


