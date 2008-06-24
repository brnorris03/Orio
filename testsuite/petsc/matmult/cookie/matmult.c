// nsz = the number of rows of matrix v1
//  sz = the number of columns of matrix v1
//  yt = the output vector
//   x = the input vector
//  v1 = the input matrix (represented in 1-d array)
// idx = the index vector (to indicate the index position of non-zero elements stored in vector x)


/*@ begin PerfTuning (  
 def build { 
   arg command = 'gcc'; 
   arg options = ''; 
 } 

 def performance_params { 
   param UI[] = [1,2,4,8]; 
 } 
 
 def input_params { 
   param NROWS[] = [4];
   param NCOLS[] = [20];
 } 
 
 def input_vars { 
   arg decl_file = 'decl_code.h';
   arg init_file = 'init_code.c'; 
 } 
 
 def performance_test_code { 
   arg skeleton_code_file = 'skeleton_code.c';  
 } 
) @*/ 


int i1,i2;
int i1t,i2t;
int nsz = NROWS;
int sz = NCOLS;

/*@ begin Loop (
transform ScalarReplace(dtype='double')
transform RegTile(loops=['i1','i2'], ufactors=[4,4])
 for (i1=0; i1<=nsz-1; i1++)
   {
     yt[i1] = 0;
     for (i2=0; i2<=sz-1; i2++)
       yt[i1] = yt[i1] + v1[i1*sz+i2] * x[idx[i2]];
   }
) @*/

for (i1=0; i1<=nsz-1; i1++) 
  {
    yt[i1] = 0;
    for (i2=0; i2<=sz-1; i2++)
      yt[i1] = yt[i1] + v1[i1*sz+i2] * x[idx[i2]];
  }

/*@ end @*/

