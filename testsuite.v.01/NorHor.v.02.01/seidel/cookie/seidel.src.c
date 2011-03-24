/*@ begin PerfTuning (          
  def build  
  {  
    arg build_command = 'icc -O3 -openmp -lm';  
  }
     
  def performance_counter           
  {  
    arg repetitions = 35;  
  } 
 
  def performance_params  
  { 
  param U1ta[] = [1,4,8,16,32];
  param U1ia[] = [1,4,8,16,32];
  param U1ja[] = [1,4,8,16,32];
  
 
    

    param U1t[] = range(1,31);
    param U1i[] = range(1,31);
    param U1j[] = range(1,31);

    constraint c2 = (U1t*U1i*U1j<=256);

    
    param SCR[] = [False,True];
    param VEC[] = [False,True];
    param PAR1t[] = [False,True];
    param PAR1i[] = [False,True];
    param PAR1j[] = [False,True];
  } 
 
  def search  
  {
    arg algorithm = 'Randomsearch';  
    arg total_runs = 20; 
  }

  def input_params  
  {
    param T[] = [500];
    param N[] = [4000];
  } 

  def input_vars
  {
    decl static double A[N][N+17] = random;
  }
) @*/  

register int i,j,k,t; 
register int c1t, c2t, c3t, c4t, c5t, c6t, c7t, c8t, c9t, c10t, c11t, c12t; 
register int newlb_c1, newlb_c2, newlb_c3, newlb_c4, newlb_c5, newlb_c6, 
  newlb_c7, newlb_c8, newlb_c9, newlb_c10, newlb_c11, newlb_c12; 
register int newub_c1, newub_c2, newub_c3, newub_c4, newub_c5, newub_c6, 
  newub_c7, newub_c8, newub_c9, newub_c10, newub_c11, newub_c12; 

/*@ begin Loop(

transform Composite(
 scalarreplace = (SCR, 'double'),
 regtile = (['t','i','j'],[U1ta,U1ia,U1ja]),
 vector = (VEC, ['ivdep','vector always'])
)
transform UnrollJam(ufactor=U1t, parallelize=PAR1t)
for (t=0; t<=T-1; t++)
  transform UnrollJam(ufactor=U1i, parallelize=PAR1i)
  for (i=1; i<=N-2; i++)
    transform UnrollJam(ufactor=U1j, parallelize=PAR1j)
    for (j=1; j<=N-2; j++)
      A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
		 + A[i][j-1] + A[i][j] + A[i][j+1]
		 + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/9.0;
) @*/
/*@ end @*/ 
/*@ end @*/ 

