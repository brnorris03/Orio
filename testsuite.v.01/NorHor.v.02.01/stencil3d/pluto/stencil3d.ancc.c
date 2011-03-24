/*@ begin PerfTuning (
  def build
  {
    arg build_command = 'icc -O3 -openmp -lm';
  }

  def performance_counter
  {
    arg repetitions = 10;
  }
  
  def performance_params
  {
    param U1i[] = range(1,31);
    param U1j[] = range(1,31);
    param U1k[] = range(1,31);

    param U2i[] = range(1,31);
    param U2j[] = range(1,31);
    param U2k[] = range(1,31);
    
    param U1ia[] = [1,4,8,16,32,64];
    param U1ja[] = [1,4,8,16,32,64];
    param U1ka[] = [1,4,8,16,32,64];


    param U2ia[] = [1,4,8,16,32,64];
    param U2ja[] = [1,4,8,16,32,64];
    param U2ka[] = [1,4,8,16,32,64];

    
    param SCR1[] = [False,True];
    param SCR2[] = [False,True];
    
    param VEC1[] = [False,True];
    param VEC2[] = [False,True];
    
    param PAR1i[]= [False, True];
    param PAR1j[]= [False, True];
    param PAR1k[]= [False, True];

    param PAR2i[]= [False, True];
    param PAR2j[]= [False, True];
    param PAR2k[]= [False, True];


  }
  
  def search
  {
   arg algorithm = 'Randomsearch';
   arg total_runs = 10;
  }

  def input_params
  {
  param N=250;
  param T=50;
  
  }

  def input_vars
  {
  decl static double a[N][N][N] = random;
  decl static double b[N][N][N] = 0;
  decl static double f1[N][N][N] = random;
  decl static double f2[N][N][N] = random;
  }

) @*/   

register int i,j,t,it,jt,tt;  

/*@ begin Loop (

transform UnrollJam(ufactor=U1t, parallelize=PAR1t)
for (t=0; t<=T-1; t++) 
  {
    
    transform Composite (  
    scalarreplace = (SCR1, 'double'), 
    regtile = (['i','j','k'],[U1ia,U1ja,U1ka]),     
    vector = (VEC1, ['ivdep','vector always'])
    )
    transform UnrollJam(ufactor=U1i, parallelize=PAR1i)
    for (i=1; i<=N-2; i++)
      transform UnrollJam(ufactor=U1j, parallelize=PAR1j)
      for (j=1; j<=N-2; j++)
        transform UnrollJam(ufactor=U1k, parallelize=PAR1k)
	for (k=1; k<=N-2; k++)
	  b[i][j][k] = f1*a[i][j][k] + f2*(a[i+1][j][k] + a[i-1][j][k] + a[i][j+1][k]
					   + a[i][j-1][k] + a[i][j][k+1] + a[i][j][k-1]);
  

    transform Composite (  
    scalarreplace = (SCR2, 'double'), 
    regtile = (['i','j','k'],[U2ia,U2ja,U2ka]),     
    vector = (VEC2, ['ivdep','vector always'])
    )   
    transform UnrollJam(ufactor=U2i, parallelize=PAR2i)
    for (i=1; i<=N-2; i++)
      transform UnrollJam(ufactor=U2j, parallelize=PAR2j)
      for (j=1; j<=N-2; j++)
        transform UnrollJam(ufactor=U2k, parallelize=PAR2k)
	for (k=1; k<=N-2; k++)
	  a[i][j][k] = b[i][j][k];
  }

) @*/
/*@ end @*/

/*@ end @*/
