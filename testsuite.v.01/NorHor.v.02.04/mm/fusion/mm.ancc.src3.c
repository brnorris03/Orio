
/*@ begin PerfTuning (        
  def build
  {
    arg build_command = 'icc -O3 -openmp ';
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
    param T1_K[] = [1,16,32,64,128,256,512];

    param T1_Ia[] = [1,64,128,256,512,1024,2048];
    param T1_Ja[] = [1,64,128,256,512,1024,2048];
    param T1_Ka[] = [1,64,128,256,512,1024,2048];



    param U_I[] = range(1,31);
    param U_J[] = range(1,31);
    param U_K[] = range(1,31);

    param SCREP[] = [False,True];
    param VEC[] = [False,True];
    param OMP[] = [False,True];

    param PAR[] = [False,True];


    param PERMUT[] = [
                       
		       (['iii'],['jjj'],['kkk'],['ii'],['jj'],['kk'],'i','j','k'),
		       (['iii'],['kkk'],['jjj'],['ii'],['kk'],['jj'],'i','j','k'),
		       (['kkk'],['jjj'],['iii'],['kk'],['jj'],['ii'],'i','j','k'),
		       (['kkk'],['iii'],['jjj'],['kk'],['ii'],['jj'],'i','j','k'),
		       (['jjj'],['kkk'],['iii'],['jj'],['kk'],['ii'],'i','j','k'),
		       (['jjj'],['iii'],['kkk'],['jj'],['ii'],['kk'],'i','j','k'),
		       
		       (['iii'],['jjj'],['kkk'],['ii'],['jj'],['kk'],'j','i','k'),
		       (['iii'],['kkk'],['jjj'],['ii'],['kk'],['jj'],'j','i','k'),
		       (['kkk'],['jjj'],['iii'],['kk'],['jj'],['ii'],'j','i','k'),
		       (['kkk'],['iii'],['jjj'],['kk'],['ii'],['jj'],'j','i','k'),
		       (['jjj'],['kkk'],['iii'],['jj'],['kk'],['ii'],'j','i','k'),
		       (['jjj'],['iii'],['kkk'],['jj'],['ii'],['kk'],'j','i','k'),

		       (['iii'],['jjj'],['kkk'],['ii'],['jj'],['kk'],'i','k','j'),
		       (['iii'],['kkk'],['jjj'],['ii'],['kk'],['jj'],'i','k','j'),
		       (['kkk'],['jjj'],['iii'],['kk'],['jj'],['ii'],'i','k','j'),
		       (['kkk'],['iii'],['jjj'],['kk'],['ii'],['jj'],'i','k','j'),
		       (['jjj'],['kkk'],['iii'],['jj'],['kk'],['ii'],'i','k','j'),
		       (['jjj'],['iii'],['kkk'],['jj'],['ii'],['kk'],'i','k','j'),

		       (['iii'],['jjj'],['kkk'],['ii'],['jj'],['kk'],'k','i','j'),
		       (['iii'],['kkk'],['jjj'],['ii'],['kk'],['jj'],'k','i','j'),
		       (['kkk'],['jjj'],['iii'],['kk'],['jj'],['ii'],'k','i','j'),
		       (['kkk'],['iii'],['jjj'],['kk'],['ii'],['jj'],'k','i','j'),
		       (['jjj'],['kkk'],['iii'],['jj'],['kk'],['ii'],'k','i','j'),
		       (['jjj'],['iii'],['kkk'],['jj'],['ii'],['kk'],'k','i','j'),

		       (['iii'],['jjj'],['kkk'],['ii'],['jj'],['kk'],'j','k','i'),
		       (['iii'],['kkk'],['jjj'],['ii'],['kk'],['jj'],'j','k','i'),
		       (['kkk'],['jjj'],['iii'],['kk'],['jj'],['ii'],'j','k','i'),
		       (['kkk'],['iii'],['jjj'],['kk'],['ii'],['jj'],'j','k','i'),
		       (['jjj'],['kkk'],['iii'],['jj'],['kk'],['ii'],'j','k','i'),
		       (['jjj'],['iii'],['kkk'],['jj'],['ii'],['kk'],'j','k','i'),

		       (['iii'],['jjj'],['kkk'],['ii'],['jj'],['kk'],'k','j','i'),
		       (['iii'],['kkk'],['jjj'],['ii'],['kk'],['jj'],'k','j','i'),
		       (['kkk'],['jjj'],['iii'],['kk'],['jj'],['ii'],'k','j','i'),
		       (['kkk'],['iii'],['jjj'],['kk'],['ii'],['jj'],'k','j','i'),
		       (['jjj'],['kkk'],['iii'],['jj'],['kk'],['ii'],'k','j','i'),
		       (['jjj'],['iii'],['kkk'],['jj'],['ii'],['kk'],'k','j','i'),

		       ];


    constraint tileI = ((T1_Ia == 1) or (T1_Ia % T1_I == 0));
    constraint tileJ = ((T1_Ja == 1) or (T1_Ja % T1_J == 0));
    constraint tileK = ((T1_Ka == 1) or (T1_Ka % T1_K == 0));
    constraint reg_capacity = (U_I*U_J + U_I*U_K + U_J*U_K <= 150);
    constraint unroll_limit = ((U_I == 1) or (U_J == 1) or (U_K == 1));

  }

  def search
  {
    arg algorithm = 'Randomsearch';
    arg total_runs = 5000;

  }
  
  def input_params
  {
    param CONT = 4000;
    param NCONT = 400;
    param M = 400;
    param N = 400;
    param K = 4000;
  }
  def input_vars
  { 
    decl static double A[M][K] = random;
    decl static double B[K][N] = random;
    decl static double C[M][N] = 0;
  }            


) @*/

int i, j, k;
int ii, jj, kk;
int iii, jjj, kkk;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*@ begin Loop(
  transform Composite(
    tile = [('i',T1_I,'ii'),('j',T1_J,'jj'),('k',T1_K,'kk'),
            (('ii','i'),T1_Ia,'iii'),(('jj','j'),T1_Ja,'jjj'),(('kk','k'),T1_Ka,'kkk')],
    permut = [PERMUT],
    unrolljam = (['i','j','k'],[U_I,U_J,U_K]),
    scalarreplace = (SCREP, 'double', 'scv_'),
    vector = (VEC, ['ivdep','vector always']),
    openmp = (OMP, 'omp parallel for private(iii,jjj,kkk,ii,jj,kk,i,j,k)')
  )
  for(i=0; i<=M-1; i++) 
    for(j=0; j<=N-1; j++)   
      for(k=0; k<=K-1; k++) 
        C[i][j] = C[i][j] + A[i][k] * B[k][j]; 

) @*/

/*@ end @*/
/*@ end @*/
