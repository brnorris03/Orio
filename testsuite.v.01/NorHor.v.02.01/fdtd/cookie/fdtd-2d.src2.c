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
    param U1j[] = range(1,31);
    param U2i[] = range(1,31);
    param U2j[] = range(1,31);
    param U3i[] = range(1,31);
    param U3j[] = range(1,31);
    param U4i[] = range(1,31);
    param U4j[] = range(1,31);


    param U1ja[] = [1,4,8,16,32,64];
    param U2ia[] = [1,4,8,16,32,64];
    param U2ja[] = [1,4,8,16,32,64];
    param U3ia[] = [1,4,8,16,32,64];
    param U3ja[] = [1,4,8,16,32,64];
    param U4ia[] = [1,4,8,16,32,64];
    param U4ja[] = [1,4,8,16,32,64];


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

    param SCREP[] = [False,True];
    param IVEC[] = [False,True];
    
    param PAR1j[]= [False, True];
    param PAR2i[]= [False, True];
    param PAR2j[]= [False, True];
    param PAR3i[]= [False, True];
    param PAR3j[]= [False, True];
    param PAR4i[]= [False, True];
    param PAR4j[]= [False, True];


  }
  
  def search
  {
   arg algorithm = 'Randomsearch';
   arg total_runs = 10;
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
  regtile = (['i'],[U1ja]),     
  vector = (IVEC, ['ivdep','vector always']))   
  transform UnrollJam(ufactor=U1j, parallelize=PAR1j)
  for (j=0; j<=ny-1; j++)
    ey[0][j] = t;

  transform Composite (  
  scalarreplace = (SCREP, 'double'), 
  regtile = (['i','j'],[U2ia,U2ja]),     
  vector = (IVEC, ['ivdep','vector always']))   
  transform UnrollJam(ufactor=U2i, parallelize=PAR2i)
  for (i=1; i<=nx-1; i++)
    transform UnrollJam(ufactor=U2j, parallelize=PAR2j)
    for (j=0; j<=ny-1; j++)
      ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]);

  transform Composite (  
  scalarreplace = (SCREP, 'double'), 
  regtile = (['i','j'],[U3ia,U3ja]),     
  vector = (IVEC, ['ivdep','vector always']))   
  transform UnrollJam(ufactor=U3i, parallelize=PAR3i)
  for (i=0; i<=nx-1; i++)
    transform UnrollJam(ufactor=U3j, parallelize=PAR3j)
    for (j=1; j<=ny-1; j++)
      ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]);

  transform Composite (  
  scalarreplace = (SCREP, 'double'), 
  regtile = (['i','j'],[U4ia,U4ja]),     
  vector = (IVEC, ['ivdep','vector always']))   
  transform UnrollJam(ufactor=U4i, parallelize=PAR4i)
  for (i=0; i<=nx-1; i++)
    transform UnrollJam(ufactor=U4j, parallelize=PAR4j)
    for (j=0; j<=ny-1; j++)
      hz[i][j]=hz[i][j]-0.7*(ex[i][j+1]-ex[i][j]+ey[i+1][j]-ey[i][j]);
}
) @*/
/*@ end @*/

/*@ end @*/
