/*@ begin PerfTuning (
  def build
  {
  arg build_command = 'icc -O3 -openmp -lm -DDYNAMIC';
  }

  def performance_counter
  {
  arg repetitions = 35;
  }  
  
  def performance_params
  {  

    param U1ia[] = [1,4,8,16,32];
    param U1ja[] = [1,4,8,16,32];
    param U1ka[] = [1,4,8,16,32];
    

    param U2ia[] = [1,4,8,16,32];
    param U2ja[] = [1,4,8,16,32];
    param U2ka[] = [1,4,8,16,32];
    

    param U1i[] = range(1,31);
    param U1j[] = range(1,31);    
    param U1k[] = range(1,31);


    param U2i[] = range(1,31);
    param U2j[] = range(1,31);    
    param U2k[] = range(1,31);



    param PAR1i[] = [False,True];
    param PAR1j[] = [False,True];
    param PAR1k[] = [False,True];

    param PAR2i[] = [False,True];
    param PAR2j[] = [False,True];
    param PAR2k[] = [False,True];
        

    param SCR1[] = [False,True];
    param VEC1[] = [False,True];
    param SCR2[] = [False,True];
    param VEC2[] = [False,True];
  
}
  
  def search
  {
  arg algorithm = 'Randomsearch';
  arg total_runs = 5000;
  }
  
  def input_params
  {
  param N[] = [4096]; 
  param alpha = 1;
  }

  def input_vars
  {
  decl static double A[N][N+20] = random;
  decl static double B[N][N+20] = random;
  }
	      

) @*/   

int i, j, k;
int ii, jj, kk;
int iii, jjj, kkk;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))


/*@ begin Loop(

transform Composite(
 scalarreplace = (SCR1, 'double'),
 regtile = (['i','j','k'],[U1ia,U1ja,U1ka]),
 vector = (VEC1, ['ivdep','vector always'])
)
transform UnrollJam(ufactor=U1i, parallelize=PAR1i)
for (i=0; i<=N-1; i++)
transform UnrollJam(ufactor=U1j, parallelize=PAR1j)
  for (j=0; j<=N-1; j++)
    {
      B[i][j] = alpha*A[i][i]*B[i][j];
      transform UnrollJam(ufactor=U1k, parallelize=PAR1k)
      for (k=i+1; k<=N-1; k++)
	B[i][j] = B[i][j] + alpha*A[i][k]*B[k][j];
    }


transform Composite(
 scalarreplace = (SCR2, 'double'),
 regtile = (['i','j','k'],[U2ia,U2ja,U2ka]),
 vector = (VEC2, ['ivdep','vector always'])
)
transform UnrollJam(ufactor=U2i, parallelize=PAR2i)
for (i=0; i<=N-1; i++)
  transform UnrollJam(ufactor=U2j, parallelize=PAR2j)
  for (j=0; j<=N-1; j++)
    transform UnrollJam(ufactor=U2k, parallelize=PAR2k)
    for (k=i+1; k<=N-1; k++)
      B[i][j] = B[i][j] + alpha*A[i][k]*B[k][j];

) @*/

/*@ end @*/
/*@ end @*/
