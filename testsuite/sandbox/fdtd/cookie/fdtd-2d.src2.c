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
    param U1[] = range(1,31);
    param U2i[] = range(1,31);
    param U2j[] = range(1,31);
    param U3i[] = range(1,31);
    param U3j[] = range(1,31);
    param U4i[] = range(1,31);
    param U4j[] = range(1,31);

    param PERM1[] = [
     ['i','j'],
     ['j','i'],
    ];
    param PERM2[] = [
     ['i','j'],
     ['j','i'],
    ];
    param PERM3[] = [
     ['i','j'],
     ['j','i'],
    ];

    param SCREP[] = [True];
    param IVEC[] = [True];
  }
  
  def search
  {
   arg algorithm = 'Randomsearch';
   arg total_runs = 5000;
  }

  def input_params
  {
  let N=250;
  param tmax[] = [500];
  param nx[] = [N];
  param ny[] = [N]; 
  }

  def input_vars
  {
  decl static double ex[nx][ny+1] = random;
  decl static double ey[nx+1][ny] = random;
  decl static double hz[nx][ny] = random;
  }
) @*/   

register int i,j,t,it,jt,tt;  

/*@ begin Loop (

for(t=0; t<=tmax-1; t++) {

  transform Composite (  
  scalarreplace = (SCREP, 'double'), 
  regtile = (['i'],[U1]),     
  vector = (IVEC, ['ivdep','vector always']))   
  for (j=0; j<=ny-1; j++)
    ey[0][j] = t;

  transform Composite (  
  permut = [PERM1], 
  scalarreplace = (SCREP, 'double'), 
  regtile = (['i','j'],[U2i,U2j]),     
  vector = (IVEC, ['ivdep','vector always']))   
  for (i=1; i<=nx-1; i++)
    for (j=0; j<=ny-1; j++)
      ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]);

  transform Composite (  
  permut = [PERM2], 
  scalarreplace = (SCREP, 'double'), 
  regtile = (['i','j'],[U3i,U3j]),     
  vector = (IVEC, ['ivdep','vector always']))   
  for (i=0; i<=nx-1; i++)
    for (j=1; j<=ny-1; j++)
      ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]);

  transform Composite (  
  permut = [PERM3], 
  scalarreplace = (SCREP, 'double'), 
  regtile = (['i','j'],[U4i,U4j]),     
  vector = (IVEC, ['ivdep','vector always']))   
  for (i=0; i<=nx-1; i++)
    for (j=0; j<=ny-1; j++)
      hz[i][j]=hz[i][j]-0.7*(ex[i][j+1]-ex[i][j]+ey[i+1][j]-ey[i][j]);
}
) @*/
/*@ end @*/

/*@ end @*/
