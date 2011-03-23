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

  param U1i[]  = range(1,31);
  param U2i[] = [1]; # MUST be 1, otherwise it generates INCORRECT results (need debugging later)
  param U2j[] = range(1,31);
  param U3i[]  = range(1,31);
  param U4i[] = range(1,31);
  param U4j[] = range(1,31);



  param U2ia[] = [1,4,8,16,32];
  param U2ja[] = [1,4,8,16,32];
  param U3ia[]  = [1,4,8,16,32];
  param U4ia[] = [1,4,8,16,32];
  param U4ja[] = [1,4,8,16,32];

  param PAR1i[] = [False,True];
  param PAR2i[] = [False,True]; 
  param PAR2j[] = [False,True]; 
  param PAR3i[] = [False,True];
  param PAR4i[] = [False,True];
  param PAR4j[] = [False,True];

  param SCR2[] = [False,True];
  param SCR3[] = [False,True];
  param SCR4[] = [False,True];

  param VEC1[] = [False,True]; 
  param VEC2[] = [False,True];
  param VEC3[] = [False,True];
  param VEC4[] = [False,True];

  }
			 
  def search 
  { 
    arg algorithm = 'Randomsearch'; 
    arg total_runs = 10;
  } 
  
  def input_params 
  {
  param N[] = [10000];
  }
  
  def input_vars
  {
  arg decl_file = 'decl.h';
  arg init_file = 'init.c';
  }
) @*/

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

int i,j,ii,jj,it,jt;

/*@ begin Loop(

transform UnrollJam(ufactor=U1i, parallelize=PAR1i)
for (i=0;i<=n-1;i++) {
  x[i]=0;
  w[i]=0;
 }


transform Composite(
 scalarreplace = (SCR2, 'double'),
 regtile = (['j','i'],[U2ia,U2ja]),
 vector = (VEC2, ['ivdep','vector always'])
)
transform UnrollJam(ufactor=U2i, parallelize=PAR2j)
for (j=0;j<=n-1;j++) {
  transform UnrollJam(ufactor=U2j, parallelize=PAR2i)
  for (i=0;i<=n-1;i++) {
    B[j*n+i]=u2[j]*v2[i]+u1[j]*v1[i]+A[j*n+i];
    x[i]=y[j]*B[j*n+i]+x[i];
  }
 }

transform Composite(
 scalarreplace = (SCR3, 'double'),
 regtile = (['i'],[U3ia]),
 vector = (VEC3, ['ivdep','vector always'])
)
transform UnrollJam(ufactor=U3i, parallelize=PAR3i)
for (i=0;i<=n-1;i++) {
  x[i]=b*x[i]+z[i];
 }


transform Composite(
 scalarreplace = (SCR4, 'double'),
 regtile = (['i','j'],[U4ia,U4ja]),
 vector = (VEC4, ['ivdep','vector always'])
)
transform UnrollJam(ufactor=U4i, parallelize=PAR4i)
for (i = 0; i <= n-1; i++) {
  transform UnrollJam(ufactor=U4j,parallelize=PAR4j)
  for (j = 0; j <= n-1; j++) {
    w[i] = w[i] + B[i*n+j]*x[j];
  }
  w[i] = a*w[i];
 }

) @*/
/*@ end @*/

/*@ end @*/



