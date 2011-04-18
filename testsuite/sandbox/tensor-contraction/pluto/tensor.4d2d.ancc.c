\
/*@ begin PerfTuning (        
  def build
  {
    arg build_command = 'icc -DDYNAMIC -fast -openmp  -lm';
  }
   
  def performance_counter         
  {
    arg repetitions = 35;
  }
  
  let VR = 100;
  let OR = 10;

  def performance_params
  {
    param U_O1[] = [1];
    param U_O2[] = [1];
    param U_OX[] = [1];
    param U_V1[] = [1];
    param U_V2[] = [1];
    param SCR1[] = [True];
    param VEC2[] = [True];
  }

  def search
  {
    arg algorithm = 'Exhaustive';
  }

  def input_params
  {
    param VSIZE = VR;
    param OSIZE = OR;
  }

  def input_vars
  {
    decl int V = VSIZE;
    decl int O = OSIZE;
    decl dynamic double A2[VSIZE][OSIZE] = random;
    decl dynamic double T[VSIZE][OSIZE][OSIZE][OSIZE] = random;
    decl dynamic double R[VSIZE][VSIZE][OSIZE][OSIZE] = 0;
  }
) @*/

int v1,v2,o1,o2,ox;
int tv1,tv2,to1,to2,tox;


/*@ begin Loop(
  transform Composite(
    scalarreplace = (SCR1,'double'),
    vector = (VEC2, ['ivdep','vector always'])
  )
  transform UnrollJam(ufactor=U_V1)
  for(v1=0; v1<=V-1; v1++) 
    transform UnrollJam(ufactor=U_V2)
    for(v2=0; v2<=V-1; v2++) 
      transform UnrollJam(ufactor=U_O1)
      for(o1=0; o1<=O-1; o1++) 
        transform UnrollJam(ufactor=U_O2)
        for(o2=0; o2<=O-1; o2++) 
	  transform UnrollJam(ufactor=U_OX)
	  for(ox=0; ox<=O-1; ox++) 
	    R[v1][v2][o1][o2] = R[v1][v2][o1][o2] + T[v1][ox][o1][o2] * A2[v2][ox];

) @*/

for(v1=0; v1<=V-1; v1++) 
  for(v2=0; v2<=V-1; v2++) 
    for(o1=0; o1<=O-1; o1++) 
      for(o2=0; o2<=O-1; o2++) 
	for(ox=0; ox<=O-1; ox++) 
	  R[v1][v2][o1][o2] = R[v1][v2][o1][o2] + T[v1][ox][o1][o2] * A2[v2][ox];

/*@ end @*/
/*@ end @*/

